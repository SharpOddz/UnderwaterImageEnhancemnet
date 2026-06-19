from google.colab import drive
import os
import shutil
import glob
import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Mount Google Drive
drive.mount('/content/drive')

#Downloading and extracting dataset
!mkdir -p /content/EUVP
!cp -n /content/drive/MyDrive/EUVP_Dataset.zip /content/EUVP_Dataset.zip
!unzip -q -n /content/EUVP_Dataset.zip -d /content/EUVP

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_SIZE = (IMAGE_HEIGHT,IMAGE_WIDTH) #Input image size
RANDOM_SEED = 23
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

dataset_dir = "/content/EUVP"
paired_subdirs = ["underwater_dark", "underwater_imagenet", "underwater_scenes"]

def find_paired_images(dataset_dir, paired_subdirs):
    pairs = []
    for subdir in paired_subdirs:
        trainA_pattern = os.path.join(
            dataset_dir, "**", "Paired", subdir, "trainA", "*.*"
        )
        trainA_files = sorted(glob.glob(trainA_pattern, recursive=True))
        for input_path in trainA_files:
            target_path = input_path.replace(
                os.sep + "trainA" + os.sep,
                os.sep + "trainB" + os.sep
            )
            if os.path.exists(target_path):
                pairs.append((input_path, target_path))
    return pairs

all_pairs = find_paired_images(dataset_dir, paired_subdirs)

#Creating the train and validation sets (the EUVP validation images are unpaired)
train_size = int(0.85 * len(all_pairs))
all_pairs = tf.random.shuffle(all_pairs, seed=RANDOM_SEED).numpy()
all_pairs = [(x.decode(), y.decode()) for x, y in all_pairs]
train_pairs = all_pairs[:train_size]
val_pairs = all_pairs[train_size:]
print(f"Training pairs: {len(train_pairs)}")
print(f"Validation pairs: {len(val_pairs)}")

#
def load_image_pair(input_path, target_path):
    input_img = tf.io.read_file(input_path)
    input_img = tf.image.decode_image(input_img, channels=3, expand_animations=False)
    input_img = tf.image.resize(input_img, IMAGE_SIZE)
    input_img = tf.cast(input_img, tf.float32) / 255.0
    target_img = tf.io.read_file(target_path)
    target_img = tf.image.decode_image(target_img, channels=3, expand_animations=False)
    target_img = tf.image.resize(target_img, IMAGE_SIZE)
    target_img = tf.cast(target_img, tf.float32) / 255.0
    return input_img, target_img

def make_dataset(pairs, training=True):
    input_paths = [p[0] for p in pairs]
    target_paths = [p[1] for p in pairs]
    ds = tf.data.Dataset.from_tensor_slices((input_paths, target_paths))
    if training:
        ds = ds.shuffle(buffer_size=len(pairs), seed=RANDOM_SEED)
    ds = ds.map(load_image_pair, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(train_pairs, training=True)
val_ds = make_dataset(val_pairs, training=False)

#Hyperparameters
epochs = 75
learning_rate = 0.0001

#The model will save to google drive or local path (adjust if not using Google Drive)
checkpoint_path = '/content/drive/MyDrive/UIE_Methods/lightweight_CNN.keras'
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

#Callbacks (early stopping, best model saving, learning rate reduction)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=10,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)

lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

#Model architecture
inputs = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
residual = layers.Conv2D(3, 3, padding="same", activation="tanh")(x)
outputs = layers.Add()([inputs, residual * 0.1])
outputs = layers.Lambda(lambda img: tf.clip_by_value(img, 0.0, 1.0))(outputs)

model = tf.keras.Model(inputs, outputs)

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss="mae"
)

history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[early_stopping, lr_reduction, checkpoint],
    verbose=1
)

model.save(checkpoint_path)
print(f"Best model successfully saved to: {checkpoint_path}")


#Plotting training history
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

#
train_pairs_by_subdir = {subdir: [] for subdir in paired_subdirs}
for pathA, pathB in all_pairs:
  for subdir in paired_subdirs:
      if subdir in pathA:
          train_pairs_by_subdir[subdir].append((pathA, pathB))
          break

#Basic UIQM Calculation
def calculate_uiqm(image):
  if image is None:
      return 0.0
  return float(np.mean(image) / 255.0 * 5.0)

#Taking an input image, enhance it using the lightweight CNN 
def model_enhance(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    enhanced_img = model.predict(img, verbose=0)[0]
    enhanced_img = (enhanced_img * 255.0).astype(np.uint8)
    enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
    return enhanced_img_bgr

#Plots the first image from each category
def plot_enhanced_images(enhancement_func, method_name="Enhanced"):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i, subdir in enumerate(paired_subdirs):
        pathA, pathB = train_pairs_by_subdir[subdir][0]
        img_unenhanced = cv2.imread(pathA)
        img_gt = cv2.imread(pathB)
        img_enhanced = enhancement_func(pathA)
        #UIQM calculation
        uiqm_unenhanced = calculate_uiqm(img_unenhanced)
        uiqm_enhanced = calculate_uiqm(img_enhanced)
        uiqm_gt = calculate_uiqm(img_gt)
        #BGR->RGB for plotting
        img_unenhanced_rgb = cv2.cvtColor(img_unenhanced, cv2.COLOR_BGR2RGB)
        img_enhanced_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)
        img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        #Unenhanced image
        axes[i, 0].imshow(img_unenhanced_rgb)
        axes[i, 0].set_title(f"Unenhanced\nUIQM: {uiqm_unenhanced:.3f}")
        axes[i, 0].set_ylabel(subdir, fontsize=12, fontweight='bold')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        #Enhanced image by lightweight CNN
        axes[i, 1].imshow(img_enhanced_rgb)
        axes[i, 1].set_title(f"{method_name}\nUIQM: {uiqm_enhanced:.3f}")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        #Ground Truth
        axes[i, 2].imshow(img_gt_rgb)
        axes[i, 2].set_title(f"Ground Truth\nUIQM: {uiqm_gt:.3f}")
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
    plt.tight_layout()
    plt.show()

plot_enhanced_images(model_enhance, "Lightweight CNN")



