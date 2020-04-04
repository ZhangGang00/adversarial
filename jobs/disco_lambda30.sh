#! /bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=mlgpu
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1

/usr/bin/nvidia-smi -L
echo "Allocate GPU cards : ${CUDA_VISIBLE_DEVICES}"

cd /hpcfs/bes/mlgpu/gang/adversarial
source setup.sh
echo 'pwd: ' $(pwd)

#python -m run.disco.train --gpu --devices=1 --verbose --optimise-classifier --config='./configs/default_disco_lambda30.json'
python -m run.disco.train --gpu --devices=1 --verbose --train-classifier --config='./configs/default_disco_lambda30.json'


