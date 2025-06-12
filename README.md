# EnhanceGAN for iPhone-to-Canon Enhancement

A from-scratch re-implementation of the concepts from the 2018 paper "Aesthetic-Driven Image Enhancement by Adversarial Learning" (EnhanceGAN), adapted for transforming low-quality iPhone photos into high-quality, DSLR-style images using the DPED dataset.

## Overview

This project implements a systematic, multi-phase training approach for image enhancement using GANs. The entire workflow is contained in a single Jupyter/Colab notebook that guides users through training and analysis phases.

## Quick Start

### Prerequisites

- Google Drive account
- Access to Google Colab
- Stable internet connection for dataset download

### Step 1: Clone/Download the Repository

1. Navigate to this GitHub repository
2. Click the green `<> Code` button
3. Select `Download ZIP`
4. Extract `enhancegan-photo-enhancement-main.zip` to your local machine

### Step 2: Download Dataset

Download the DPED dataset subset:

- **Link**: [DPED Small Dataset](https://drive.google.com/file/d/1WqOVQqhbsDL41x56NlDz_2filRmgmt2F/view?usp=drivesdk)
- Download `dped_small.zip` to your local machine

**About the DPED Small Dataset**: This is a truncated version of the original DPED dataset (available at https://people.ee.ethz.ch/~ihnatova/) specifically created for lightweight, fast training in environments like Google Colab. The smaller size significantly speeds up download and unzipping times while maintaining representative samples for educational and prototyping purposes.

### Step 3: Set Up Google Drive Structure

Create the following folder structure in your Google Drive:

```
/My Drive/
└── enhancegan-photo-enhancement/          # Main project folder
    ├── enhance_GAN_training_and_analysis_final.ipynb
    ├── dped_small.zip
    ├── data_prep.py
    ├── datasets.py
    └── requirements.txt
```

**Upload these 5 files to your Google Drive folder:**

1. `enhance_GAN_training_and_analysis_final.ipynb`
2. `data_prep.py`
3. `datasets.py`
4. `requirements.txt`
5. `dped_small.zip` (downloaded from Step 2)

### Step 4: Run in Google Colab

1. In Google Drive, navigate to the `enhancegan-photo-enhancement` folder
2. Right-click on `enhance_GAN_training_and_analysis_final.ipynb`
3. Select `Open with` → `Google Colaboratory`
4. Run cells sequentially starting from the first cell

## Workflow Overview

The notebook is organized into distinct phases that should be executed in order:

### Phase 1: Setup

- **Part 1**: Environment & Data Setup
- **Part 2**: Data Preparation (executes `data_prep.py`)
- **Part 3**: Dataset and DataLoaders setup

### Phase 2: Training (Parts 4-6)

| Phase       | Description                  | Output                                               |
| ----------- | ---------------------------- | ---------------------------------------------------- |
| **Phase 1** | Baseline U-Net training      | `generator_phase1.pth`                               |
| **Phase 2** | Add CurveBlock + fine-tuning | `generator_phase2.pth`                               |
| **Phase 3** | Full WGAN-GP framework       | `generator_phase3.pth`<br>`discriminator_phase3.pth` |

### Phase 3: Analysis

- Comprehensive evaluation and analysis
- Quantitative metrics (PSNR/SSIM)
- Visual comparisons
- Performance reports

## Generated Directory Structure

The notebook automatically creates these folders during execution:

```
enhancegan-photo-enhancement/
├── data/                           # Processed train/test splits
├── checkpoints/                    # Saved model weights (.pth files)
└── paper_results_final/           # Analysis figures and reports
```

## Technical Details

### Model Architecture

- **Generator**: U-Net based architecture with progressive enhancement
- **Discriminator**: WGAN-GP framework for realistic image generation
- **Loss Functions**: Combination of content loss, adversarial loss, and perceptual loss

### Training Strategy

1. **Phase 1**: Content-focused training using MSE loss
2. **Phase 2**: Enhanced feature learning with CurveBlock
3. **Phase 3**: Adversarial training for photorealistic results

## Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- Visual quality assessment through generated comparison images

## Requirements

The notebook automatically installs required dependencies via `requirements.txt`. Key libraries include:

- PyTorch
- torchvision
- PIL (Pillow)
- NumPy
- Matplotlib

## Notes

- **Runtime**: Training phases may take several hours depending on Colab resources
- **GPU**: Recommended for faster training (Colab Pro for better GPU access)
- **Storage**: Ensure sufficient Google Drive storage for datasets and outputs

## Dataset Information

This project uses a subset of the DPED (DSLR Photo Enhancement Dataset) originally introduced in "DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks" (ICCV 2017). The original dataset contains over 22K photos captured synchronously from three smartphones (iPhone 3GS, BlackBerry Passport, Sony Xperia Z) and one Canon 70D DSLR camera.

**Original DPED Dataset**: https://people.ee.ethz.ch/~ihnatova/

## References

- Original EnhanceGAN paper: "Aesthetic-Driven Image Enhancement by Adversarial Learning" (2018)
- DPED Dataset: "DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks" (ICCV 2017)
