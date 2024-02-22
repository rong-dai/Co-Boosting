python fl_pretrain.py \
--dataset="cifar10" --partition="dir"  --beta=0.6 --seed=42 --num_users=10 \
--model="cnn" \
--local_lr=0.01 --local_ep=100 \
--sigma 0.0 \
--batch_size 128
