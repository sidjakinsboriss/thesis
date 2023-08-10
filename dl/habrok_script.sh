#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000

module load Python/3.10.4-GCCcore-11.3.0
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

pip install keras-tuner
pip install iterative-stratification
pip install matplotlib
pip install gensim
pip install --upgrade transformers

# Run the necessary script, e.g. python -m dl.classifiers.bert.training_bert
