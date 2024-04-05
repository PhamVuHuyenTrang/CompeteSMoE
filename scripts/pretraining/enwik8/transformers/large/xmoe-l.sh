mkdir -p checkpoints/enwik8/transformers-l/xmoe

args="
--data /lustre/scratch/client/vinai/users/trangpvh1/repo/CompteSMoE/pretraining/enwik8 \
--base_arch transformer \
--architecture sgsgsgsgsgsgsgsgsgsgsgsg \
--gate_name xmoe \
--nlayers 8 \
--hid-sz 528 \
--inner-hid-sz 528 \
--nheads 12 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 6000 \
--niter 120 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint checkpoints/enwik8/transformers-l/xmoe/xmoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8