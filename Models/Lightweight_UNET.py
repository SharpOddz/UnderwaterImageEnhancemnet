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

#Input image size
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_SIZE = (IMAGE_HEIGHT,IMAGE_WIDTH)

#Seed setting for reproducibility
RANDOM_SEED = 23
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

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

#Train, validation, and test sets with stratification
train_pairs = []
val_pairs = []
test_pairs = []

for subdir in paired_subdirs:
    subdir_pairs = [p for p in all_pairs if subdir in p[0]]
    random.seed(RANDOM_SEED)
    random.shuffle(subdir_pairs)

    n = len(subdir_pairs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_pairs.extend(subdir_pairs[:n_train])
    val_pairs.extend(subdir_pairs[n_train:n_train+n_val])
    test_pairs.extend(subdir_pairs[n_train+n_val:])

random.seed(RANDOM_SEED)
random.shuffle(train_pairs)
random.seed(RANDOM_SEED)
random.shuffle(val_pairs)
random.seed(RANDOM_SEED)
random.shuffle(test_pairs)

print(f"Training pairs: {len(train_pairs)}")
print(f"Validation pairs: {len(val_pairs)}")
print(f"Testing pairs: {len(test_pairs)}")

def load_image_pair(input_path, target_path):
    input_img = tf.io.read_file(input_path)
    input_img = tf.image.decode_image(input_img, channels=3, expand_animations=False)
    input_img.set_shape([None, None, 3])
    input_img = tf.image.resize(input_img, IMAGE_SIZE, antialias=True)
    input_img = tf.cast(input_img, tf.float32) / 255.0

    target_img = tf.io.read_file(target_path)
    target_img = tf.image.decode_image(target_img, channels=3, expand_animations=False)
    target_img.set_shape([None, None, 3])
    target_img = tf.image.resize(target_img, IMAGE_SIZE, antialias=True)
    target_img = tf.cast(target_img, tf.float32) / 255.0

    return input_img, target_img

def make_dataset(pairs, training=True):
    input_paths = [p[0] for p in pairs]
    target_paths = [p[1] for p in pairs]

    ds = tf.data.Dataset.from_tensor_slices((input_paths, target_paths))

    if training:
        ds = ds.shuffle(
            buffer_size=len(pairs),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(load_image_pair, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=training)
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(train_pairs, training=True)
val_ds = make_dataset(val_pairs, training=False)
test_ds = make_dataset(test_pairs, training=False)

#CNN Model Architecture
#Goal is for the model to be lightweight and real-time inference capable
#Hyperparameters
epochs = 100
learning_rate = 0.0001
RESIDUAL_SCALE = 0.2

#The model will save to google drive or local path (adjust if not using Google Drive)
best_model_path = '/content/drive/MyDrive/UIE_Methods/lightweight_UNET_best.keras'
final_model_path = '/content/drive/MyDrive/UIE_Methods/lightweight_UNET_final.keras'
os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

#Callbacks (early stopping, best model saving, learning rate reduction)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=10,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_path,
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

#Model architecture - Lightweight U-Net
def conv_block(x, filters):
    #would be worth investigating to see if the image quality is improved without batch normalization
    shortcut = x
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

inputs = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))

#Encoder
e1 = conv_block(inputs, 16)
p1 = layers.MaxPooling2D(2)(e1)
e2 = conv_block(p1, 32)
p2 = layers.MaxPooling2D(2)(e2)
e3 = conv_block(p2, 64)
p3 = layers.MaxPooling2D(2)(e3)

#Bottleneck
b = conv_block(p3, 128)
b = layers.Conv2D(128, 3, padding="same", dilation_rate=2, use_bias=False)(b)
b = layers.BatchNormalization()(b)
b = layers.Activation("relu")(b)
b = layers.Conv2D(128, 3, padding="same", dilation_rate=4, use_bias=False)(b)
b = layers.BatchNormalization()(b)
b = layers.Activation("relu")(b)

#Decoder
d1 = layers.UpSampling2D(2, interpolation="bilinear")(b)
d1 = layers.Concatenate()([d1, e3])
d1 = conv_block(d1, 64)
d2 = layers.UpSampling2D(2, interpolation="bilinear")(d1)
d2 = layers.Concatenate()([d2, e2])
d2 = conv_block(d2, 32)
d3 = layers.UpSampling2D(2, interpolation="bilinear")(d2)
d3 = layers.Concatenate()([d3, e1])
d3 = conv_block(d3, 16)

delta = layers.Conv2D(3, 1, padding="same", activation="tanh")(d3)

@tf.keras.utils.register_keras_serializable()
def residual_output(z):
    input_img, delta = z
    return tf.clip_by_value(input_img + RESIDUAL_SCALE * delta, 0.0, 1.0)

outputs = layers.Lambda(residual_output)([inputs, delta])

model = tf.keras.Model(inputs, outputs)

model.summary()

#Custom losses and metrics
@tf.keras.utils.register_keras_serializable()
def gradient_loss(y_true, y_pred):
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)

    return (
        tf.reduce_mean(tf.abs(dy_true - dy_pred)) +
        tf.reduce_mean(tf.abs(dx_true - dx_pred))
    )

@tf.keras.utils.register_keras_serializable()
def uie_loss(y_true, y_pred):
    y_pred_clipped = tf.clip_by_value(y_pred, 0.0, 1.0)

    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred_clipped, max_val=1.0))
    grad = gradient_loss(y_true, y_pred)

    return mae + 0.2 * ssim_loss + 0.05 * grad

@tf.keras.utils.register_keras_serializable()
def psnr_metric(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

@tf.keras.utils.register_keras_serializable()
def ssim_metric(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=uie_loss,
    metrics=[psnr_metric, ssim_metric]
)

history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    callbacks=[early_stopping, lr_reduction, checkpoint],
    verbose=1
)

model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")
print(f"Best validation model saved to: {best_model_path}")

import matplotlib.pyplot as plt

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

test_pairs_by_subdir = {subdir: [] for subdir in paired_subdirs}
for pathA, pathB in test_pairs:
  for subdir in paired_subdirs:
      if subdir in pathA:
          test_pairs_by_subdir[subdir].append((pathA, pathB))
          break

import cv2
import numpy as np

#Full UIQM Calculation
def uicm(img):
    b, g, r = cv2.split(img)
    rg = r.astype(np.float32) - g.astype(np.float32)
    yb = 0.5 * (r.astype(np.float32) + g.astype(np.float32)) - b.astype(np.float32)
    rg_mean = np.mean(rg)
    yb_mean = np.mean(yb)
    rg_var = np.var(rg)
    yb_var = np.var(yb)
    uicm_val = -0.0268 * np.sqrt(rg_var**2 + yb_var**2) + 0.1586 * np.sqrt(rg_mean**2 + yb_mean**2)
    return uicm_val

def uism(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.hypot(sobelx, sobely)
    return np.mean(sobel)

def uiconm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.std(gray)

def calculate_uiqm(image):
    if image is None:
        return 0.0
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753
    val_uicm = uicm(image)
    val_uism = uism(image)
    val_uiconm = uiconm(image)
    uiqm = c1 * val_uicm + c2 * val_uism + c3 * val_uiconm
    return float(uiqm)

#Load the best validation model for evaluation
print(f"Loading best model from {best_model_path}...")
model = tf.keras.models.load_model(best_model_path)

print("\n=== Best Model Test Evaluation ===")
test_results = model.evaluate(test_ds, verbose=1)
for name, value in zip(model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")

#Taking an input image, enhance it using the lightweight CNN
def model_enhance(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    enhanced_img = model.predict(img, verbose=0)[0]
    enhanced_img = np.clip(enhanced_img, 0.0, 1.0)
    enhanced_img = (enhanced_img * 255.0).astype(np.uint8)
    enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
    return enhanced_img_bgr

#Plots the first image from each category
def plot_enhanced_images(enhancement_func, method_name="Enhanced"):
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i, subdir in enumerate(paired_subdirs):
        pathA, pathB = test_pairs_by_subdir[subdir][0]
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
        #Enhanced image by lightweight UNET
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

#Plotting function call using the UNET model
plot_enhanced_images(model_enhance, "Lightweight U-Net")


'''
import pandas as pd
from tqdm import tqdm
from IPython.display import display
import tensorflow as tf
import random

# Define splits for easy iteration
splits = {
    'Train': train_pairs,
    'Val': val_pairs,
    'Test': test_pairs
}

# Set a sample limit to speed up evaluation (evaluating 11k images can take ~10+ minutes)
# Set this to None if you want to evaluate the entire dataset for your final thesis results.
MAX_SAMPLES_PER_GROUP = None

# Initialize a dedicated random number generator for reproducible evaluation sampling
eval_rng = random.Random(RANDOM_SEED)

records = []

for split_name, pairs in splits.items():
    # Group pairs by category
    grouped_pairs = {cat: [] for cat in paired_subdirs}
    for pathA, pathB in pairs:
        for cat in paired_subdirs:
            if cat in pathA:
                grouped_pairs[cat].append((pathA, pathB))
                break

    for cat, cat_pairs in grouped_pairs.items():
        # Sample if necessary
        if MAX_SAMPLES_PER_GROUP and len(cat_pairs) > MAX_SAMPLES_PER_GROUP:
            cat_pairs = eval_rng.sample(cat_pairs, MAX_SAMPLES_PER_GROUP)

        for pathA, pathB in tqdm(cat_pairs, desc=f"{split_name} - {cat}"):
            img_unenhanced = cv2.imread(pathA)
            img_gt = cv2.imread(pathB)
            img_enhanced = model_enhance(pathA)

            # Resize img_gt to match the fixed size of img_enhanced for metric calculations
            img_gt_resized = cv2.resize(img_gt, IMAGE_SIZE)

            # UIQM calculations
            # Note: For consistency in UIQM, you might also want to resize img_unenhanced to IMAGE_SIZE
            # if calculate_uiqm expects a consistent input size.
            uiqm_u = calculate_uiqm(img_unenhanced)
            uiqm_e = calculate_uiqm(img_enhanced)
            uiqm_gt = calculate_uiqm(img_gt) # UIQM calculated on original size of ground truth

            # Convert to RGB for standard metrics
            img_gt_rgb = cv2.cvtColor(img_gt_resized, cv2.COLOR_BGR2RGB) # Use the resized ground truth for paired metrics
            img_enhanced_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)

            # Convert to tensors and normalize to [0, 1] for TF metric functions
            gt_tf = tf.convert_to_tensor(img_gt_rgb, dtype=tf.float32) / 255.0
            enh_tf = tf.convert_to_tensor(img_enhanced_rgb, dtype=tf.float32) / 255.0

            # Calculate paired metrics
            mae = tf.reduce_mean(tf.abs(gt_tf - enh_tf)).numpy()
            psnr = tf.image.psnr(gt_tf, enh_tf, max_val=1.0).numpy()
            ssim = tf.image.ssim(gt_tf, enh_tf, max_val=1.0).numpy()

            records.append({
                'Split': split_name,
                'Category': cat,
                'MAE': mae,
                'PSNR': psnr,
                'SSIM': ssim,
                'UIQM Input': uiqm_u,
                'UIQM Enhanced': uiqm_e,
                'UIQM GT': uiqm_gt
            })

# Create DataFrame and compute averages
df_results = pd.DataFrame(records)
summary_df = df_results.groupby(['Split', 'Category']).mean().reset_index()

print("\n=== Comprehensive Evaluation Metrics ===")
display(summary_df)
'''


