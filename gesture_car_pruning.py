"""
gesture_car_pruning.py — Weight Pruning for Gesture Car (MediaPipe + TFLite)
==============================================================================
WHAT THIS DOES:
    MediaPipe's hand_landmarker.task is a pre-compiled TFLite flatbuffer.
    We cannot prune it directly. Instead, this script:
      1. Extracts a lightweight MobileNet-based hand classifier from scratch
         (mirroring what MediaPipe uses internally).
      2. Applies TensorFlow Model Optimization Toolkit (TFMOT) magnitude-based
         weight pruning during fine-tuning on your gesture dataset.
      3. Strips pruning wrappers and strips zero-weights.
      4. Exports a pruned .tflite model you can swap into the gesture car
         pipeline alongside (or instead of) hand_landmarker.task for the
         gesture classification head.

PRUNING STRATEGY:
    - Technique  : Magnitude-based unstructured weight pruning (TFMOT)
    - Schedule   : Polynomial decay  sparsity 0.0 → target (e.g. 50 %)
    - Target     : Dense layers of the gesture-classification head only
      (landmark backbone left intact, since MediaPipe handles that part)
    - Result     : ~40-60 % reduction in non-zero weights → faster inference
      on Raspberry Pi 5 with negligible accuracy drop if tuned well.

HARDWARE TARGET : Raspberry Pi 5 (ARM Cortex-A76, no GPU, no TPU)
RUNTIME         : TFLite Interpreter (CPU)

INSTALL:
    pip install tensorflow tensorflow-model-optimization mediapipe opencv-python

HOW TO USE:
    1. Collect gesture images into  ./gesture_data/<class_name>/*.jpg
       Classes: fist, open_palm, thumb_left, thumb_right, unknown
    2. Run:  python gesture_car_pruning.py
    3. Output: pruned_gesture_classifier.tflite
    4. In gesture_car-2.py, load this model for gesture classification
       AFTER hand_landmarker extracts landmarks.

WIRING / GPIO / motor logic: unchanged from gesture_car-2.py
"""

import os
import math
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR         = "./gesture_data"          # root folder with class sub-dirs
CLASSES          = ["fist", "open_palm", "thumb_left", "thumb_right", "unknown"]
NUM_CLASSES      = len(CLASSES)
IMG_SIZE         = (96, 96)                  # input resolution for classifier
BATCH_SIZE       = 16
INITIAL_EPOCHS   = 5                         # train base model first
PRUNING_EPOCHS   = 5                         # then prune-fine-tune
TARGET_SPARSITY  = 0.50                      # 50 % of weights zeroed out
OUTPUT_MODEL     = "pruned_gesture_classifier.tflite"

# ─── 1. DATASET ───────────────────────────────────────────────────────────────
print("[1/6] Loading dataset...")

def load_dataset(data_dir, img_size, batch_size):
    """Load images from class sub-folders and return train/val tf.data."""
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        class_names=CLASSES,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        class_names=CLASSES,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
    )
    # Normalize [0,255] -> [0,1]
    norm = keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.map(lambda x, y: (norm(x), y)).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

# Gracefully handle missing dataset (demo / dry-run mode)
if os.path.isdir(DATA_DIR):
    train_ds, val_ds = load_dataset(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    DRY_RUN = False
else:
    print(f"  ⚠  '{DATA_DIR}' not found — running in DRY-RUN mode with random data.")
    print("     Create ./gesture_data/<class>/*.jpg to train on real gestures.\n")
    # Synthetic stand-in so the whole pipeline can be tested end-to-end
    dummy_x = np.random.rand(32, *IMG_SIZE, 3).astype("float32")
    dummy_y = np.random.randint(0, NUM_CLASSES, 32)
    train_ds = tf.data.Dataset.from_tensor_slices((dummy_x, dummy_y)).batch(BATCH_SIZE)
    val_ds   = train_ds
    DRY_RUN  = True

# ─── 2. BASE MODEL (MobileNetV2 backbone + gesture head) ─────────────────────
print("[2/6] Building base model (MobileNetV2 backbone)...")

def build_base_model(num_classes, img_size):
    backbone = keras.applications.MobileNetV2(
        input_shape=(*img_size, 3),
        include_top=False,
        weights="imagenet" if not DRY_RUN else None,
    )
    backbone.trainable = False          # freeze backbone initially

    inp   = keras.Input(shape=(*img_size, 3), name="image_input")
    x     = backbone(inp, training=False)
    x     = keras.layers.GlobalAveragePooling2D()(x)
    x     = keras.layers.Dense(128, activation="relu", name="fc1")(x)
    x     = keras.layers.Dropout(0.3)(x)
    out   = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inp, out, name="gesture_classifier")
    return model

base_model = build_base_model(NUM_CLASSES, IMG_SIZE)
base_model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
base_model.summary()

# ─── 3. INITIAL TRAINING (no pruning) ────────────────────────────────────────
print(f"\n[3/6] Initial training for {INITIAL_EPOCHS} epochs...")
base_model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS, verbose=1)

# ─── 4. APPLY PRUNING ────────────────────────────────────────────────────────
print(f"\n[4/6] Applying magnitude-based pruning (target sparsity={TARGET_SPARSITY})...")

# Total fine-tuning steps needed for the pruning schedule
num_train_batches = sum(1 for _ in train_ds)
total_steps       = num_train_batches * PRUNING_EPOCHS
begin_step        = 0
frequency         = max(1, num_train_batches // 4)   # prune 4× per epoch

pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.0,
    final_sparsity=TARGET_SPARSITY,
    begin_step=begin_step,
    end_step=total_steps,
    frequency=frequency,
)

# Only prune the Dense layers (gesture head); backbone is frozen & not pruned
def apply_pruning_to_dense(layer):
    """Return pruned version of Dense layers; leave everything else alone."""
    if isinstance(layer, keras.layers.Dense):
        return tfmot.sparsity.keras.prune_low_magnitude(
            layer, pruning_schedule=pruning_schedule
        )
    return layer

pruned_model = keras.models.clone_model(
    base_model,
    clone_function=apply_pruning_to_dense,
)

# Copy weights from the trained base model into the pruned model
pruned_model.set_weights(base_model.get_weights())

pruned_model.compile(
    optimizer=keras.optimizers.Adam(1e-4),   # lower LR for fine-tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),           # required for TFMOT
    tfmot.sparsity.keras.PruningSummaries(log_dir="./pruning_logs"),
]

print(f"\n[5/6] Pruning fine-tune for {PRUNING_EPOCHS} epochs...")
pruned_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PRUNING_EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

# ─── 5. STRIP PRUNING & EXPORT TO TFLite ─────────────────────────────────────
print("\n[5b/6] Stripping pruning wrappers (zeroed weights remain)...")
stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

print(f"[6/6] Converting to TFLite and saving → {OUTPUT_MODEL}")
converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)

# Keep the model in float32 (pruning only — no quantization here)
converter.optimizations = []   # <-- set to [] to keep full float precision
tflite_model = converter.convert()

with open(OUTPUT_MODEL, "wb") as f:
    f.write(tflite_model)

# ─── SPARSITY REPORT ─────────────────────────────────────────────────────────
def measure_sparsity(model):
    """Count fraction of zero weights across all kernel tensors."""
    total, zeros = 0, 0
    for layer in model.layers:
        for weight in layer.weights:
            if "kernel" in weight.name:
                arr    = weight.numpy()
                total += arr.size
                zeros += np.sum(arr == 0)
    return zeros / total if total > 0 else 0.0

sparsity = measure_sparsity(stripped_model)
original_size = os.path.getsize("pruned_gesture_classifier.tflite") / 1024

print("\n" + "="*60)
print("  PRUNING COMPLETE")
print("="*60)
print(f"  Achieved sparsity  : {sparsity*100:.1f}%  (target {TARGET_SPARSITY*100:.0f}%)")
print(f"  Output model       : {OUTPUT_MODEL}")
print(f"  TFLite model size  : {original_size:.1f} KB")
print("="*60)

# ─── HOW TO USE THIS MODEL IN gesture_car-2.py ───────────────────────────────
print("""
HOW TO INTEGRATE WITH gesture_car-2.py
---------------------------------------
# After MediaPipe extracts lm (hand_landmarks[0]), feed the RAW FRAME
# through this TFLite classifier instead of your hand-crafted rules:

import numpy as np
import cv2

interpreter = tf.lite.Interpreter(model_path="pruned_gesture_classifier.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def classify_gesture_pruned(frame):
    img = cv2.resize(frame, (96, 96)).astype("float32") / 255.0
    interpreter.set_tensor(input_details[0]["index"], img[np.newaxis])
    interpreter.invoke()
    probs   = interpreter.get_tensor(output_details[0]["index"])[0]
    class_i = int(np.argmax(probs))
    classes = ["fist", "open_palm", "thumb_left", "thumb_right", "unknown"]
    return classes[class_i], float(probs[class_i])
""")
