from google.colab import drive
import os
import shutil
import glob
import cv2
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import cv2

#Mount Google Drive
drive.mount('/content/drive')

#Downloading and extracting dataset
!mkdir -p /content/EUVP
!cp -n /content/drive/MyDrive/EUVP_Dataset.zip /content/EUVP_Dataset.zip
!unzip -q -n /content/EUVP_Dataset.zip -d /content/EUVP

#Base directory and subdirectories
dataset_dir = './'
paired_subdirs = ['underwater_dark', 'underwater_imagenet', 'underwater_scenes']
train_pairs_by_subdir = {subdir: [] for subdir in paired_subdirs}
val_images_by_subdir = {subdir: [] for subdir in paired_subdirs}

train_pairs = []
val_images = []
for subdir in paired_subdirs:
    #Finding all images within the corresponding subfolder
    trainA_pattern = os.path.join(dataset_dir, '**', 'Paired', subdir, 'trainA', '*.*')
    trainA_files = glob.glob(trainA_pattern, recursive=True)
    for pathA in trainA_files:
        pathB = pathA.replace(os.sep + 'trainA' + os.sep, os.sep + 'trainB' + os.sep)
        if os.path.exists(pathB):
            train_pairs.append((pathA, pathB))
            train_pairs_by_subdir[subdir].append((pathA, pathB))
    #Validation images
    val_pattern = os.path.join(dataset_dir, '**', 'Paired', subdir, 'validation', '*.*')
    found_val = glob.glob(val_pattern, recursive=True)
    val_images.extend(found_val)
    val_images_by_subdir[subdir].extend(found_val)

#Verification there are in total: 11,435 training pairs and 1970 validation images
#print(f"Total paired training images (trainA & trainB) found: {len(train_pairs)}")
#print(f"Total validation images found: {len(val_images)}")

#Verification that the datasets contain the correct number of training pairs and validation images
#print("\nTraining pair counts per dataset:")
#for subdir in paired_subdirs:
#    print(f"{subdir}: {len(train_pairs_by_subdir[subdir])} training pairs, {len(val_images_by_subdir[subdir])} validation images")


#Test pairs
test_pairs = []
#Recursive glob to find the corresponding subfolders
test_inp_pattern = os.path.join(dataset_dir, '**', '*test_sample*', 'Inp', '*.*')
test_inp_files = glob.glob(test_inp_pattern, recursive=True)
for path_inp in test_inp_files:
    path_gtr = path_inp.replace(os.sep + 'Inp' + os.sep, os.sep + 'GTr' + os.sep)
    if os.path.exists(path_gtr):
        test_pairs.append((path_inp, path_gtr))

#Verification that there are 515 test pairs
#print(f"Total test pairs found: {len(test_pairs)}")


#Unpaired Images
unpaired_trainA = []
unpaired_trainB = []
unpaired_val = []
#Recursive glob to find the corresponding subfolders
unpaired_trainA_pattern = os.path.join(dataset_dir, '**', 'Unpaired', 'trainA', '*.*')
unpaired_trainB_pattern = os.path.join(dataset_dir, '**', 'Unpaired', 'trainB', '*.*')
unpaired_val_pattern = os.path.join(dataset_dir, '**', 'Unpaired', 'validation', '*.*')
#Glob + filtering out a few duplicates in the trainA folder
unpaired_trainA = [f for f in glob.glob(unpaired_trainA_pattern, recursive=True) if '(' not in f]
unpaired_trainB = glob.glob(unpaired_trainB_pattern, recursive=True)
unpaired_val = glob.glob(unpaired_val_pattern, recursive=True)
#Verification that there are 3195 trainA, 3140 trainB, 330 validation images
#print(f"Total unpaired poor quality (trainA) images: {len(unpaired_trainA)}")
#print(f"Total unpaired good quality (trainB) images: {len(unpaired_trainB)}")
#print(f"Total unpaired validation images: {len(unpaired_val)}")

#Image quality metrics
def uicm(img):
    b, g, r = cv2.split(img)
    R = r.astype(np.float32)
    G = g.astype(np.float32)
    B = b.astype(np.float32)
    RG = R - G
    YB = (R + G) / 2 - B
    mu_RG, mu_YB = np.mean(RG), np.mean(YB)
    var_RG, var_YB = np.var(RG), np.var(YB)
    uicm_value = -0.0268 * np.sqrt(mu_RG**2 + mu_YB**2) + 0.1586 * np.sqrt(var_RG + var_YB)
    return uicm_value

def uism_uiconm(img):
    #Simplified versions using Sobel and Michelson contrast proxies for quick calculation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Sharpness proxy (Edge magnitude)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    uism_value = np.mean(magnitude) / 100.0
    #Contrast proxy (standard deviation of grayscale)
    uiconm_value = np.std(gray) / 100.0
    return uism_value, uiconm_value

def calculate_uiqm(img):
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uicm_val = uicm(img)
    uism_val, uiconm_val = uism_uiconm(img)
    uiqm = (c1 * uicm_val) + (c2 * uism_val) + (c3 * uiconm_val)
    return uiqm

#Classical Methods (White Balance, CLAHE, Gamma Correction, Histogram Equalization, Retinex Corrections)

#White Balance (Gray World Algorithm)
def gray_world_white_balance(img_path):
    img = cv2.imread(img_path)
    b, g, r = cv2.split(img)
    b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
    total_avg = (b_avg + g_avg + r_avg) / 3
    b_gain = total_avg / b_avg
    g_gain = total_avg / g_avg
    r_gain = total_avg / r_avg
    b = cv2.convertScaleAbs(b, alpha=b_gain)
    g = cv2.convertScaleAbs(g, alpha=g_gain)
    r = cv2.convertScaleAbs(r, alpha=r_gain)
    return cv2.merge((b, g, r))

#CLAHE
#Clip Limit and tile grid size can be chnaged depending on the exact type of UW image but a generic 2.0 and (8,8) was chosen
def apply_clahe(img_path):
    img = cv2.imread(img_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

#Gamma Correction
def apply_gamma_correction(img_path, gamma=1.5):
    img = cv2.imread(img_path)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

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
        #Unenhanced
        axes[i, 0].imshow(img_unenhanced_rgb)
        axes[i, 0].set_title(f"Unenhanced\nUIQM: {uiqm_unenhanced:.3f}")
        axes[i, 0].set_ylabel(subdir, fontsize=12, fontweight='bold')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        #Enhanced
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

#Plotting function call, adjust the function call as needed
plot_enhanced_images(apply_gamma_correction, "Gamma Correction")


#The comments below need to be adjusted, this is not the best/correct way to call this function
#You need to pass in the enhancement function and then give it a method name (name for the pivot table)
#If you don't want to use any enhancement then simply put "None" for enhancement_func
def evaluate_dataset_uiqm(enhancement_func=apply_clahe, method_name="CLAHE"):
    results = []
    print(f"\nEvaluating UIQM: {method_name} ...")
    for subdir in paired_subdirs:
        train_uiqm = []
        print(f"Processing Train - {subdir}")
        for pathA, _ in tqdm(train_pairs_by_subdir[subdir]):
            if enhancement_func is not None:
                img = enhancement_func(pathA)
            else:
                img = cv2.imread(pathA)
                
            if img is not None:
                train_uiqm.append(calculate_uiqm(img))
        
        avg_train_uiqm = np.mean(train_uiqm) if train_uiqm else 0
        val_uiqm = []
        print(f"Processing Validation - {subdir}")
        for path in tqdm(val_images_by_subdir[subdir]):
            if enhancement_func is not None:
                img = enhancement_func(path)
            else:
                img = cv2.imread(path)
            if img is not None:
                val_uiqm.append(calculate_uiqm(img))
        avg_val_uiqm = np.mean(val_uiqm) if val_uiqm else 0
        results.append({
            'Category': subdir,
            'Split': 'Train',
            'Average UIQM': avg_train_uiqm
        })
        results.append({
            'Category': subdir,
            'Split': 'Validation',
            'Average UIQM': avg_val_uiqm
        })
    #Pivot table of UIQM scores across all categories
    df_uiqm = pd.DataFrame(results)
    pivot_df = df_uiqm.pivot(index='Category', columns='Split', values='Average UIQM')
    print(f"\nAverage UIQM Scores - {method_name} (Pivot Table)")
    display(pivot_df)
    return pivot_df

pivot_table = evaluate_dataset_uiqm()
