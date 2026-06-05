# Underwater Image Enhancement (UIE)
Exploration of ~11 UIE methods on the Enhancing Underwater Visual Perception (EUVP) dataset. The purpose of this project is to determine whether UIE improves downstream tasks such as swimmer pose estimation, temporal stroke modelling, and swimmer action quality assessment. Each UIE method will be evaluated on both image quality metrics and pose/action metrics.

Each method is evaluated on the following image quality metrics: PSNR, SSIM, UIQM, UCIQE. Each method is evaluated on the following pose/action metrics: PCK, OKS, keypoint confidence, temporal keypoint jitter, stroke phase classification accuracy, action quality score error.

## UIE Methods
The following UIE methods are evaluated:
- Raw image (baseline)
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
