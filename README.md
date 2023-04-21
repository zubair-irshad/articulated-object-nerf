# Articulated Object Neural Radiance Field


# Preliminary Qualitative Results:

<div style="display:flex;">
  <img src="demo/gif1.gif" width="33.33%">
  <img src="demo/gif2.gif" width="33.33%">
  <img src="demo/gif3.gif" width="33.33%">
</div>

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

* Single Instance Overfitting training
``` python run.py --dataset_name sapien --root_dir /experiments/zubair/datasets/sapien_single_scene --exp_name sapien_single_scene_vanilla --exp_type vanilla --img_wh 640 480 --white_back --batch_size 1 --num_gpus 1 --num_gpus 8  --num_epochs 100```


* Single Instance Articulated Overfitting

``` CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 python run.py --dataset_name sapien_multi --root_dir /experiments/zubair/datasets/sapien_single_scene_art --exp_name sapien_single_scene_articulated --exp_type vanilla_autodecoder --img_wh 320 240 --white_back --batch_size 1 --num_gpus 7 ```

# :key: Evaluation

* Single Scene Overfitting:

``` CUDA_VISIBLE_DEVICES=0 python run.py --dataset_name sapien --root_dir /experiments/zubair/datasets/sapien_single_scene --exp_name sapien_single_scene_vanilla --exp_type vanilla --img_wh 640 480 --white_back --batch_size 1 --num_gpus 1 --run_eval --render_name sapien_test_highres ```

* Single Insatnce Articulation Overfitting
``` CUDA_VISIBLE_DEVICES=0 python /home/ubuntu/zubair/articulated-object-nerf/run.py --dataset_name sapien_multi --root_dir /experiments/zubair/datasets/sapien_single_scene_art --exp_name sapien_single_scene_articulated --exp_type vanilla_autodecoder --img_wh 320 240 --white_back --batch_size 1 --N_max_objs 1 --run_eval --render_name single_image_train_degs_interpolation2 ```


# :key: Generate Sapien Dataset
* Coming Soon

