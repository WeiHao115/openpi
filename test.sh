#!/usr/bin/env bash
set -euo pipefail
cd /home/k202/openpi/openpi
source /opt/ros/noetic/setup.bash
if [ -f /home/k202/UR10/devel/setup.bash ]; then
  source /home/k202/UR10/devel/setup.bash
fi
source ~/anaconda3/etc/profile.d/conda.sh
conda activate OPENPI
CHECKPOINT_DIR="/home/8TDisk/0709model_c16_openpi/pi05_lerobot_C16_ethernet_cable/plug_pi05_pytorch/30000"
python UR10_deploy/run_UR10_bywei_angle.py \
  --config-name pi05_lerobot_C16_ethernet_cable \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --device cuda \
  --task "Plug the Ethernet cable into the Ethernet port" \
  --action-chunk-size 8 \
  --num-inference-steps 16 \
  --gopro-device-id 6

# Insert the Ethernet cable plug into the network port slot