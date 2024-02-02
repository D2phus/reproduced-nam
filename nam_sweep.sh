#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=700M
#SBATCH --job-name=nam-array-hardcoded
#SBATCH --output=nam-array-hardcoded_%a.out
#SBATCH --array=0-5

case $SLURM_ARRAY_TASK_ID in
   0)  ACT_TYPE='gelu'
       NUM_BASIS=64
   ;;
   1)  ACT_TYPE='gelu'
       NUM_BASIS=1024
   ;;
   2)  ACT_TYPE='relu'  
       NUM_BASIS=64
   ;;
   3)  ACT_TYPE='relu'  
       NUM_BASIS=1024
   ;;
   4)  ACT_TYPE='exu'  
       NUM_BASIS=64
   ;;
   5)  ACT_TYPE='exu'  
       NUM_BASIS=1024 
   ;;
esac

srun python nam_sweep.py --activation=$ACT_TYPE --num_basis_functions=$NUM_BASIS