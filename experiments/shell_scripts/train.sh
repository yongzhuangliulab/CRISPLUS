#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=16
#SBATCH --mem 200G
#SBATCH -p optimal
#SBATCH -A optimal 
# nvidia-smi

echo export OMP_NUM_THREADS=4
echo export OPENBLAS_NUM_THREADS=4
echo export MKL_NUM_THREADS=4
echo export VECLIB_MAXIMUM_THREADS=4
echo export NUMEXPR_NUM_THREADS=4
echo export NUMBA_NUM_THREADS=4


# echo mkdir ../results/OP3_complete_unseen_drug
# for sp in split1 split2 split3 split4; do
#     for sd in 1327 1337 1347; do
#         for MMD in 0.1; do
#             echo \
#                 mkdir ../results/OP3_complete_unseen_drug/OP3_complete_unseen_drug_${sp}_${sd};
#         done;
#     done;
# done;


for sp in split1 split2 split3 split4; do
    for sd in 1327 1337 1347; do
        echo \
            rm ../results/OP3_complete_unseen_drug/OP3_complete_unseen_drug_${sp}_${sd}/*;
    done;
done;


# for sp in split1 split2 split3 split4; do
#     for sd in 1327 1337 1347; do
#         for MMD in 0.00001; do
#             echo \
#                 nohup \
#                 python ../../CRISPLUS/train_script.py \
#                 --config ../configs/OP3_complete_unseen_drug.yaml \
#                 --split $sp \
#                 --seed $sd \
#                 --savedir ../results/OP3_complete_unseen_drug/OP3_complete_unseen_drug_${sp}_${sd} \
#                 --MMD $MMD \
#                 --celltype_co 1 \
#                 \> ../results/OP3_complete_unseen_drug/OP3_complete_unseen_drug_${sp}_${sd}/output.log \
#                 2\>\&1 \
#                 \&;
#         done;
#     done;
# done;


# echo mkdir ../results/OP3_complete
# for sp in split1 split2 split3 split4; do
#     for sd in 1327 1337 1347; do
#         for MMD in 0.1; do
#             echo \
#                 mkdir ../results/OP3_complete/OP3_complete_${sp}_${sd};
#         done;
#     done;
# done;


for sp in split1 split2 split3 split4; do
    for sd in 1327 1337 1347; do
        echo \
            rm ../results/OP3_complete/OP3_complete_${sp}_${sd}/*;
    done;
done;


# for sp in split1 split2 split3 split4; do
#     for sd in 1327 1337 1347; do
#         for MMD in 0.0001; do
#             echo \
#                 nohup \
#                 python ../../CRISPLUS/train_script.py \
#                 --config ../configs/OP3_complete.yaml \
#                 --split $sp \
#                 --seed $sd \
#                 --savedir ../results/OP3_complete/OP3_complete_${sp}_${sd} \
#                 --MMD $MMD \
#                 --celltype_co 1 \
#                 \> ../results/OP3_complete/OP3_complete_${sp}_${sd}/output.log \
#                 2\>\&1 \
#                 \&;
#         done;
#     done;
# done;


# for sd in 1327 1337 1347; do
#     for sp in split split2 split3; do
#         for MMD in 0.1; do
#             echo \
#                 nohup \
#                 python ../../CRISP/train_script.py \
#                 --config ../configs/nips.yaml \
#                 --split $sp \
#                 --seed $sd \
#                 --savedir ../results/nips/nips_${sp}_${sd} \
#                 --MMD $MMD \
#                 \> ../results/nips/nips_${sp}_${sd}/output.log \
#                 2\>\&1 \
#                 \&;
#         done;
#     done;
# done;

 
# for sd in 1327 1337 1347; do
#     for sp in split split2 split3; do
#         for MMD in 0.0001; do
#             echo \
#                 nohup \
#                 python ../../CRISP/train_script.py \
#                 --config ../configs/sci.yaml \
#                 --split $sp \
#                 --seed $sd \
#                 --savedir ../results/sci/sci_${sp}_${sd} \
#                 --MMD $MMD \
#                 \> ../results/sci/sci_${sp}_${sd}/output.log \
#                 2\>\&1 \
#                 \&;
#         done;
#     done;
# done;
