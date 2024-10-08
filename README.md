
# Image Translation with CycleGAN - PyTorch

This repository provides an implementation of a CycleGAN model for image-to-image translation using PyTorch. CycleGAN allows you to learn mappings between two different domains without requiring paired examples. This can be used for tasks like translating sketches into real images, converting photos into artistic styles, and much more.

## Features

- **CycleGAN Architecture**: Implements the CycleGAN model using two generators and two discriminators for unpaired image translation.
- **Customizable Architectures**: Select from various generator architectures (e.g., ResNet, U-Net).
- **Checkpointing**: Save and load model checkpoints during training for easy resumption or evaluation.
- **Visualization**: Generate and visualize translated images using Google Colab or local paths.
- **Web Deployment**: Instructions to deploy the model on a web interface for user interaction.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Testing](#testing)
- [Visualization](#visualization)
- [Web Deployment](#web-deployment)
- [References](#references)

## Installation

### Prerequisites

- Python 3.6 or higher
- PyTorch 1.0 or higher
- CUDA (optional, for GPU support)

### Steps to Install

1. Clone the repository:

    ```bash
    git clone https://github.com/Haule9-2/ImgToImg_pytorch.git
    cd ImgToImg_pytorch
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your dataset following the instructions in the [Dataset Preparation](#dataset-preparation) section.

## Dataset Preparation

Prepare your dataset by organizing images into two folders representing the two domains you wish to translate between. For example, for a sketch-to-photo translation task:

```
dataset/
    trainA/         # Folder for sketches
        sketch1.jpg
        sketch2.jpg
        ...
    trainB/         # Folder for real images
        image1.jpg
        image2.jpg
        ...
```

You can use any dataset suitable for your specific translation task, as long as you follow the unpaired format.

## Training

To train the CycleGAN model, run the following command in your terminal:

```bash
python train.py --dataroot ./dataset --name experiment_name --model cycle_gan --netG resnet_9blocks --niter 200 --niter_decay 100
```

### Parameters

- `--dataroot`: Path to the dataset directory containing `trainA` and `trainB`.
- `--name`: Experiment name to save results.
- `--model`: Model type, set to `cycle_gan`.

You can customize other hyperparameters in the `train.py` file as needed.

## Testing

After training is complete, you can test the model using the following command:

```bash
python test.py --dataroot ./dataset --name experiment_name --model cycle_gan 
```

This command will generate translated images and save them in the specified output directory.

## Visualization

After training your CycleGAN model, you can visualize the translated images generated by the model. To make it easier for users to visualize the results, you can access the web application that I’ve developed for this purpose.

### Access the Web Application

You can visualize the translated images using my web app. Simply follow these steps:

1. **Visit the Web App**: Go to the following link to access the web interface:
   (https://github.com/Haule9-2/image-translation-cyclegan-web-app.git).

2. **Upload Images**: Use the interface to upload images you want to translate. 

3. **View Translations**: After uploading, the translated images will be displayed directly in your browser.

## Web Deployment

Deploy your trained CycleGAN model on a web interface to allow users to upload images and see translated results. Follow these steps:

1. **Web Interface**: Build a web interface using React.js for the frontend that allows users to upload images for translation.
   
2. **Backend Integration**: Create a backend using a framework like Flask or FastAPI to handle image uploads and perform translations, returning results to the user.

For detailed instructions on deploying your model, check out my [Web Deployment GitHub Repository](https://github.com/Haule9-2/image-translation-cyclegan-web-app.git).

## References

- [CycleGAN Paper](https://arxiv.org/abs/1703.10593)
- [Official PyTorch Implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
