# Underwater Image Enhancement (UIE)
Exploration of ~11 UIE methods on the Enhancing Underwater Visual Perception (EUVP) dataset. The purpose of this project is to determine whether UIE improves downstream tasks such as swimmer pose estimation, temporal stroke modelling, and swimmer action quality assessment. Each UIE method will be evaluated on both image quality metrics and pose/action metrics. The dataset for pose estimation is TBD and will be decided upon completion of exploration of UIE methods. A single 3D pose estimation model will be chosen for the pose evaluation (to be determined as of 6/05/2026). 

Each method is evaluated on the following image quality metrics: PSNR, SSIM, UIQM, UCIQE. Each method is evaluated on the following pose/action metrics: PCK, OKS, keypoint confidence, temporal keypoint jitter, stroke phase classification accuracy, action quality score error.

## UIE Methods
The following UIE methods are evaluated:
- Classical methods (non-deep learning baseline): white balance, CLAHE, gamma correction, histogram equalization, retinex corrections
- Lightweight CNN
- Lightweight U-Net
- Generative Adversarial Networks (GAN)
- Physics Based Image Restoration
- Frequency/Wavelet/Fourier Transformation
- Frequency Fusion
- Transfomer/Hybrid Attention Restoration
- Diffusion Restoration
- Flow/Rectifed Flow/Efficient Generative Restoration
- Task-aware UIE
- Temporal Consistency Restoration

## Requirement

## Dataset
The EUVP dataset can be download at: https://www.kaggle.com/datasets/pamuduranasinghe/euvp-dataset?resource=download

## UIQM Results (No Enhancements)
This pivot table represents the average UIQM scores of all the train/val images in EUVP with no enhancement. This serves as a baseline.

<img width="368" height="217" alt="image" src="https://github.com/user-attachments/assets/4b0043b0-19cb-4bd1-b6e3-aceaab67d7e3" />

## White Balance UIQM Results

<img width="331" height="152" alt="image" src="https://github.com/user-attachments/assets/9db5b8b0-7cff-4d04-bfa7-d719fa69d871" />


