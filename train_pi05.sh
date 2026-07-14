cd /home/k202/openpi/openpi
# uv run scripts/compute_norm_stats.py \
#   --config-name pi05_lerobot_C16_ethernet_cable

uv run /home/k202/openpi/openpi/scripts/train_pytorch.py \
  pi05_lerobot_C16_ethernet_cable \
  --exp-name plug_pi05_pytorch \
  --batch-size 4 \
  --pytorch-weight-path /home/k202/openpi_pi05 \
  --checkpoint-base-dir /home/8TDisk/0713model_c16_openpi \
  --num-train-steps 30000 \
  --save-interval 2000 \
  --overwrite