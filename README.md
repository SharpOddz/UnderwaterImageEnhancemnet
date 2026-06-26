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

<img width="1467" height="1189" alt="image" src="https://github.com/user-attachments/assets/47d8f15d-ab41-4f19-ac23-13b76734a433" />


## CLAHE UIQM Results
A generic was used, clip limit of 2.0 and tile grid size of (8,8)

<img width="335" height="158" alt="image" src="https://github.com/user-attachments/assets/742e18dd-35d0-44bf-b841-dbdde7b70dfe" />

<img width="1467" height="1189" alt="image" src="https://github.com/user-attachments/assets/b00e2e41-428f-4a27-8ec1-899416f8ce3f" />

## Gamma Correction UIQM Results

<img width="332" height="159" alt="image" src="https://github.com/user-attachments/assets/1d0ab09e-105c-4d0c-90a1-0291e8cbbbd0" />

<img width="1467" height="1189" alt="image" src="https://github.com/user-attachments/assets/e9018472-e25d-483e-b3ac-a08bb0bcc01f" />

## Lightweight CNN Results

<img width="1443" height="1189" alt="image" src="https://github.com/user-attachments/assets/9c2a7979-3ec9-4f37-bb7a-34a1a1f51853" />

## Lightweight U-Net Results

<img width="1443" height="1189" alt="image" src="https://github.com/user-attachments/assets/fe38aa01-3073-42a0-a24b-a1ace64f4359" />

## Lightweight Transformer Results

<img width="1443" height="1189" alt="image" src="https://github.com/user-attachments/assets/e8ceb306-534b-4a76-9600-1b05211d7c52" />

