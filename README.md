# Articulated Object Neural Radiance Field

# :computer: Overview
Experimental Repo for Modelling Neural Radiance Field for Articulated Objects. Currently Supported Experiments:

- Sapien Dataset (Single Instance Overfitting)

- Sapien Dataset (Single Instance Articulated Overfitting)

- Sapien Dataset (Single Instance Auto-Encoder Articulated NeRF)

- Future: Sapien Dataset (Single Instance Auto-Decoder Articulated NeRF)


# :computer: Installation

## Hardware

* OS: Ubuntu 18.04
* NVIDIA GPU with **CUDA>=10.2** (tested with 1 RTX2080Ti)

## Software

* Clone this repo by `git clone --recursive https://github.com/zubair-irshad/articulated-object-nerf`
* Python>=3.7 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n ao-nerf python=3.7` to create a conda environment and activate it by `conda activate nerf_pl`)
* Python libraries
    * Install core requirements by `pip install -r requirements.txt`
    
# :key: Training

``` python run.py --dataset_name sapien --root_dir /experiments/zubair/datasets/sapien_single_scene --exp_name sapien_single_scene_vanilla --exp_type vanilla --img_wh 640 480 --white_back --batch_size 1 --num_gpus 1 --num_gpus 8  --num_epochs 100```


# :key: Evaluation

* Single Scene Overfitting:

``` CUDA_VISIBLE_DEVICES=0 python run.py --dataset_name sapien --root_dir /experiments/zubair/datasets/sapien_single_scene --exp_name sapien_single_scene_vanilla --exp_type vanilla --img_wh 640 480 --white_back --batch_size 1 --num_gpus 1 --run_eval --render_name sapien_test_highres ```

# :key: Generate Sapien Dataset
* Coming Soon

