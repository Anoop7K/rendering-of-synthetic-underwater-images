# 🌊 Synthetic Underwater Image Generation using Conditional GANs (cGAN)

This project implements a **Conditional Generative Adversarial Network (cGAN)** to generate **synthetic underwater images** from clean in-air ground truth images, conditioned on **Jerlov water type maps**.

## 📊 Dataset Structure
The dataset uses a CSV-based mapping system to maintain alignment between image triplets:

File Mapping: A CSV file contains the filenames that map corresponding images across all three directories

**Image Directories**:

**input/:** Contains target underwater images

**inputJerlov/:** Contains Jerlov water type conditioning maps

**groundTruth/:** Contains clean ground truth images

**Note**: The actual image data is confidential and not included in this repository. The CSV file ensures proper alignment between the corresponding images across these directories without exposing the sensitive data.

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone <your-repo-url>
cd rendering-of-synthetic-underwater-images
pip install -r requirements.txt
```

## 🚀 Training

To train the cGAN model:

```bash
python train.py --version v1 --learning_rate 0.0002
```

The training process saves:
- Model checkpoints
- Training and validation loss metrics
- Training logs
- Generated sample images during training

## 🖼️ Inference

Generate synthetic underwater images using a trained model:

```bash
python inference.py --model_path results/v1/Final_Model.pth \
                   --input datasets/groundTruth/sample.png \
                   --jerlov datasets/inputJerlov/sample.png \
                   --output results/v1/samples/generated_image.png
```

Generated outputs are saved in the results directory with appropriate timestamps.

## 📈 Evaluation

Compute quantitative metrics for a trained generator against the dataset's real underwater images:

```bash
python evaluate.py --generator results/checkpoints/G_epoch200.pth \
                    --discriminator results/checkpoints/D_epoch200.pth
```

Reports and saves (as JSON):
- **PSNR** – pixel-level fidelity to the target underwater image
- **SSIM** – structural similarity
- **L1 / MAE** – the same pixel loss used during training
- **Perceptual loss** – VGG16 feature-space distance
- **Adversarial loss** – how well the generator fools the discriminator (requires `--discriminator`)
- **Discriminator loss & accuracy** – how well D separates real from fake (requires `--discriminator`)

## 🧠 Model Architecture

The implementation uses:
- **Generator**: U-Net style architecture with encoder-decoder structure and skip connections
- **Discriminator**: PatchGAN classifier that operates on image patches
- **Conditional GAN**: Both generator and discriminator receive Jerlov water type maps as conditioning input

## 📌 Key Features

- Conditional image generation based on water type parameters
- Configurable training parameters via YAML configuration
- CSV-based dataset management for flexible data organization
- Comprehensive logging and visualization utilities
- Support for training resumption from checkpoints
- Multiple loss functions including adversarial, L1, and perceptual losses

## 🔒 Data Confidentiality

This repository maintains the confidentiality of the underwater image dataset by:

- Using a CSV mapping system instead of including actual images
- Providing a clear structure for organizing proprietary data
- Including data preparation scripts that work with the CSV mapping approach