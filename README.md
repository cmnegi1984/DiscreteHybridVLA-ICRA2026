
# DiscreteHybridVLA — Team 28 Kinetix AI
## ICRA 2026 AIRoA VLA Pipeline Competition

# DiscreteHybridVLA — Team 28 Kinetix AI
## ICRA 2026 AIRoA VLA Pipeline Competition

**Authors:**
- Chandra Mohan Singh Negi — AI/Digitalization Leader, Siemens Technology & Services | M.Tech AI/CSE, IIT Jodhpur
- Prof. Gaurav Harit — Dept. of CSE, IIT Jodhpur

---

## Model Summary

| Property | Value |
|---|---|
| Architecture | DiscreteHybridVLA v6 |
| State input | observation.state (8-DoF) |
| Action output | action.absolute (8-DoF) |
| Task conditioning | task_index embedding |
| Codebook | K=512 motion primitives |
| Final training loss | 0.2705 |
| Training data | existing episodes + public_tasks/task6911 |
| Evaluation tasks | coffee bottle, box, mug relocation |

---

## Checkpoint Location
```
s3://airoa-icra-team-28/submission_v5/checkpoint_team28_v6.pt
```

---

## Reproduction Steps
```bash
mkdir -p /workspace/kinetix_ai
cp -r ./* /workspace/kinetix_ai/
export MODEL_CHECKPOINT=/workspace/kinetix_ai/checkpoint_team28_v6.pt
export DEVICE=cuda
export CODEBOOK_K=512
export STATE_DIM=8
export ACTION_DIM=8
./RUN-DOCKER-CONTAINER.sh up
./RUN-DOCKER-CONTAINER.sh shell
roslaunch hsr_policy_client hsr_policy_client.launch \
  checkpoint:=/workspace/kinetix_ai/checkpoint_team28_v6.pt \
  device:=cuda
```

---

## Dependencies
```
torch==2.2.0
torchvision==0.17.0
transformers==4.40.0
ftfy regex tqdm
git+https://github.com/openai/CLIP.git
boto3 pandas pyarrow numpy
opencv-python-headless accelerate Pillow
```

---

## Submission History

| Round | Bucket | Checkpoint | Loss |
|---|---|---|---|
| Round 1 | submission_v1 | checkpoint_team28_v1.pt | — |
| Round 2 | submission_v2 | checkpoint_team28_v3_clip.pt | — |
| Round 3 | submission_v3 | checkpoint_team28_v4.pt | 0.2886 |
| Round 3 (updated) | submission_v4 | checkpoint_team28_v5.pt | 0.0868 |
| **Round 3 (final)** | **submission_v5** | **checkpoint_team28_v6.pt** | **0.2705** |
