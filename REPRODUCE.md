# Team 28 Kinetix AI — Exact Reproduction Steps

## Setup
mkdir -p /workspace/kinetix_ai
cp -r ./* /workspace/kinetix_ai/
cd /workspace/kinetix_ai

## Environment variables
export MODEL_CHECKPOINT=/workspace/kinetix_ai/checkpoint_team28_v6.pt
export DEVICE=cuda
export CODEBOOK_K=512
export STATE_DIM=8
export ACTION_DIM=8

## Run Docker container
./RUN-DOCKER-CONTAINER.sh up
./RUN-DOCKER-CONTAINER.sh shell

## Launch HSR policy
roslaunch hsr_policy_client hsr_policy_client.launch \
  checkpoint:=/workspace/kinetix_ai/checkpoint_team28_v6.pt \
  device:=cuda
