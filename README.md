
# DiscreteHybridVLA — Team 28 Kinetix AI
## ICRA 2026 AIRoA VLA Pipeline Competition

**Authors:** Chandra Mohan Singh Negi (Siemens / IIT Jodhpur) · Prof. Gaurav Harit (IIT Jodhpur)

## Model Summary
- Architecture: DiscreteHybridVLA (discrete action tokenisation + confidence-gated shared autonomy)
- State input: observation.state (8-DoF)
- Action output: action.absolute (8-DoF)
- Task conditioning: task_index embedding
- Codebook: K=512 motion primitives
- Final training loss: 0.2886 (CE)
- Tasks: coffee bottle, box, mug relocation

## Checkpoint
Available at: s3://airoa-icra-team-28/submission_v3/checkpoint_team28_v4.pt

## Reproduction Steps
```bash
mkdir -p /workspace/kinetix_ai
cp -r ./* /workspace/kinetix_ai/
export MODEL_CHECKPOINT=/workspace/kinetix_ai/checkpoint_team28_v4.pt
export DEVICE=cuda
export CODEBOOK_K=512
export STATE_DIM=8
export ACTION_DIM=8
./RUN-DOCKER-CONTAINER.sh up
./RUN-DOCKER-CONTAINER.sh shell
roslaunch hsr_policy_client hsr_policy_client.launch \
  checkpoint:=/workspace/kinetix_ai/checkpoint_team28_v4.pt \
  device:=cuda
```

## Dependencies (Dockerfile)
- torch==2.2.0
- torchvision==0.17.0
- transformers==4.40.0
- ftfy, regex, tqdm
- git+https://github.com/openai/CLIP.git
- boto3, pandas, pyarrow, numpy
- opencv-python-headless, accelerate, Pillow
