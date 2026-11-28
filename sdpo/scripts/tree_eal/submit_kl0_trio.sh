#!/bin/bash
# Submit MT-Bench mix prep, then Tree EAL / Chain EAL / Pure KL (KL=0 ablate), 2 GPUs each.
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
SLURM_DIR=$REPO/sdpo/slurm/tree_eal
mkdir -p /scratch/ll5914/logs/tree_eal

PREP=$(sbatch --parsable "$SLURM_DIR/prep_mix_sg_gsm_he_mt.sbatch")
echo "PREP=$PREP"

J_TREE=$(sbatch --parsable --dependency=afterok:${PREP} \
  --job-name=kl0_tree_eal \
  --export=ALL,METHOD=tree_eal,TAG=kl0_tree_eal_mix4k5,REPO=$REPO \
  "$SLURM_DIR/train_kl0_trio.sbatch")
J_CHAIN=$(sbatch --parsable --dependency=afterok:${PREP} \
  --job-name=kl0_chain_eal \
  --export=ALL,METHOD=chain_eal,TAG=kl0_chain_eal_mix4k5,REPO=$REPO \
  "$SLURM_DIR/train_kl0_trio.sbatch")
J_KL=$(sbatch --parsable --dependency=afterok:${PREP} \
  --job-name=kl0_pure_kl \
  --export=ALL,METHOD=kl,TAG=kl0_pure_kl_mix4k5,REPO=$REPO \
  "$SLURM_DIR/train_kl0_trio.sbatch")

echo "TREE=$J_TREE CHAIN=$J_CHAIN KL=$J_KL"
squeue -u "$USER" -o '%.18i %.20j %.12P %.8T %.10M %.25R' | head -20
