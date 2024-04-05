mkdir -p checkpoints/wikitext-103/glam-s/xmoe

args="
--data /lustre/scratch/client/vinai/users/trangpvh1/repo/CompteSMoE/pretraining/wikitext-103 \
--base_arch glam \
--architecture sgsfsgsfsgsf \
--gate_name xmoe \
--nlayers 3 \
--hid-sz 144 \
--inner-hid-sz 144 \
--nheads 8 \
--block-sz 512 \
--attn-span 2048 \
--dropout 0.1 \
--load_balance 0.01 \
--optim adam \
--lr 0.0007 \
--lr-warmup 3000 \
--niter 60 \
--batch-sz 48 \
--batch-split 2 \
--nbatches 1000 \
--checkpoint checkpoints/wikitext-103/glam-s/xmoe/xmoe.pt \
"

echo "Training ..."
python train.py $args

echo "Evaluation ..."
python train.py $args --full-eval-mode --batch-sz 8