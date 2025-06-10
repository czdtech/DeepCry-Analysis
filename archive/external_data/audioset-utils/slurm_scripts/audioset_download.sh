#!/bin/sh

#SBATCH	-A standby

#standby
#jgmakin-n
#training
#debug

#SBATCH --constraint=F|G|I|K|D|B|H|J|C|N 

# High Mem GPUs: F|G|I|K|D
# very Fast GPUs: F|K
# Fast GPUs: B|D
# Slow GPUs: E

#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --mem=0

#--mem=0 is no longer supported, 
# If you want to use the entire node's memory,
# you can submit the job with the --exclusive option
# otherwise specify memory explicitly e.g. --mem=20G

hostname
# SLURM_NTASKS: The total number of tasks requested for the job.
# SLURM_NTASKS_PER_NODE: The number of tasks per node
echo "total tasks: "
echo $SLURM_NTASKS
echo "total tasks per node: "
echo $SLURM_NTASKS_PER_NODE
echo "cpus per task: "
echo $SLURM_CPUS_PER_TASK
echo "Cuda visible devices"
echo $CUDA_VISIBLE_DEVICES

NUMBA_DISABLE_INTEL_SVML=1

module purge
module load anaconda/2020.11-py38
module load use.own
module load conda-env/wav2letter-py3.8.5
# module load gcc/9.3.0
#module load conda-env/wav2letter_pretrained-py3.8.5


srun python ../scripts/download_subset_audioset.py "$@" -n $SLURM_CPUS_PER_TASK
