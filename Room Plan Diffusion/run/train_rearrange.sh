#!/bin/bash
# Please update the path to your own environment in config.yaml and following arguments befrore running the script
cd ./scripts

exp_dir="your experiment directory"

#### bedrooms
config="../config/rearrange/diffusion_bedrooms_instancond_lat32_v_rearrange.yaml"
exp_name="diffusion_bedrooms_instancond_lat32_v_rearrange"
python train_diffusion.py $config $exp_dir --experiment_tag $exp_name  --with_wandb_logger

#### diningrooms
config="../config/rearrange/diffusion_diningrooms_instancond_lat32_v_rearrange.yaml"
exp_name="diffusion_diningrooms_instancond_lat32_v_rearrange"
python train_diffusion.py $config $exp_dir --experiment_tag $exp_name  --with_wandb_logger

#### livingrooms
config="../config/rearrange/diffusion_livingrooms_instancond_lat32_v_rearrange.yaml"
exp_name="diffusion_livingrooms_instancond_lat32_v_rearrange"
python train_diffusion.py $config $exp_dir --experiment_tag $exp_name  --with_wandb_logger