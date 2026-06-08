from google.colab import drive
import os
import shutil
import glob

#Mount Google Drive
drive.mount('/content/drive')

#Downloading and extracting dataset
!cp /content/drive/MyDrive/EUVP_Dataset.zip .
!unzip -q -o /content/EUVP_Dataset.zip

#Train Images (Broken down into three categories: udnerwater_dark, underawter_imagenet, underwater_scenes)
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



#Classical Methods (White Balance, CLAHE, Gamma Correction, Histogram Equalization, Retinex Corrections)
