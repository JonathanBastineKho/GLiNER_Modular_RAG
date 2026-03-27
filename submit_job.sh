#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4        # Matches your DataLoader num_workers
#SBATCH --mem=40G                # Increased RAM to prevent OOM crashes
#SBATCH --job-name=6127_train
#SBATCH --time=06:00:00
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err

module load anaconda
eval "$(conda shell.bash hook)"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda activate 6127env

export LD_LIBRARY_PATH=/home/msai/ngch0136/.conda/envs/6127env/lib:$LD_LIBRARY_PATH

# pip install -r requirements.txt
# pip install -e .

python train_rq.py
