## Training/test Tips
#### Training/test options
Please see `options/train_options.py` and `options/base_options.py` for the training flags; see `options/test_options.py` and `options/base_options.py` for the test flags. There are some model-specific flags as well, which are added in the model files, such as `--lambda_A` option in `model/cycle_gan_model.py`. The default values of these options are also adjusted in the model files.
#### CPU/GPU (default `--gpu_ids 0`)
Please set`--gpu_ids -1` to use CPU mode; set `--gpu_ids 0,1,2` for multi-GPU mode. You need a large batch size (e.g., `--batch_size 32`) to benefit from multiple GPUs.

#### Fine-tuning/resume training
To fine-tune a pre-trained model, or resume the previous training, use the `--continue_train` flag. The program will then load the model based on `epoch`. By default, the program will initialize the epoch count as 1. Set `--epoch_count <int>` to specify a different starting epoch count.


#### Prepare your own datasets for CycleGAN
You need to create two directories to host images from domain A `/path/to/data/trainA` and from domain B `/path/to/data/trainB`. Then you can train the model with the dataset flag `--dataroot /path/to/data`. Optionally, you can create hold-out test datasets at `/path/to/data/testA` and `/path/to/data/testB` to test your model on unseen images.


Create folder `/path/to/data` with subdirectories `A` and `B`. `A` and `B` should each have their own subdirectories `train`, `val`, `test`, etc. In `/path/to/data/A/train`, put training images in style A. In `/path/to/data/B/train`, put the corresponding images in style B. Repeat same for other data splits (`val`, `test`, etc).

Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., `/path/to/data/A/train/1.jpg` is considered to correspond to `/path/to/data/B/train/1.jpg`.

Once the data is formatted this way, call:
```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```

This will combine each pair of images (A,B) into a single image file, ready for training.


#### About image size
 Since the generator architecture in CycleGAN involves a series of downsampling / upsampling operations, the size of the input and output image may not match if the input image size is not a multiple of 4. As a result, you may get a runtime error because the L1 identity loss cannot be enforced with images of different size. Therefore, we slightly resize the image to become multiples of 4 even with `--preprocess none` option. For the same reason, `--crop_size` needs to be a multiple of 4.

#### Training/Testing with high res images
CycleGAN is quite memory-intensive as four networks (two generators and two discriminators) need to be loaded on one GPU, so a large image cannot be entirely loaded. In this case, we recommend training with cropped images. For example, to generate 1024px results, you can train with `--preprocess scale_width_and_crop --load_size 1024 --crop_size 360`, and test with `--preprocess scale_width --load_size 1024`. This way makes sure the training and test will be at the same scale. At test time, you can afford higher resolution because you don’t need to load all networks.

#### Training/Testing with rectangular images
CycleGAN can work for rectangular images. To make them work, you need to use different preprocessing flags. Let's say that you are working with `360x256` images. During training, you can specify `--preprocess crop` and `--crop_size 256`. This will allow your model to be trained on randomly cropped `256x256` images during training time. During test time, you can apply the model on `360x256` images with the flag `--preprocess none`.

There are practical restrictions regarding image sizes for each generator architecture. For `unet256`, it only supports images whose width and height are divisible by 256. For `unet128`, the width and height need to be divisible by 128. For `resnet_6blocks` and `resnet_9blocks`, the width and height need to be divisible by 4.

#### About loss curve
Unfortunately, the loss curve does not reveal much information in training GANs, and CycleGAN is no exception. To check whether the training has converged or not, we recommend periodically generating a few samples and looking at them.

