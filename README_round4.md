
# DiscreteHybridVLA — Team 28 Kinetix AI
## ICRA 2026 AIRoA VLA Pipeline Competition

**Authors:** Chandra Mohan Singh Negi (Siemens / IIT Jodhpur) · Prof. Gaurav Harit (IIT Jodhpur)

## Root cause of previous failures (Rounds 1-3)

Previous submissions failed because we were using PyTorch while the evaluation
pipeline expected JAX/openpi. We also had CLIP as a dependency which caused
ModuleNotFoundError on startup. Round 4 fixes both issues completely.

## Package Structure

src/discrete_hybrid_vla/
├── __init__.py      ← exports DiscreteHybridVLA, HSRAdapter
├── model.py         ← DiscreteHybridVLA (PyTorch)
└── adapter.py       ← HSRAdapter with predict() and reset()

## Reproduction Steps

```bash
mkdir -p /workspace/kinetix_ai
cp -r ./* /workspace/kinetix_ai/
export MODEL_CHECKPOINT=/workspace/kinetix_ai/checkpoint_team28_v6.pt
export DEVICE=cuda
./RUN-DOCKER-CONTAINER.sh up
./RUN-DOCKER-CONTAINER.sh shell
roslaunch hsr_policy_client hsr_policy_client.launch
```

## Model Summary

| Property | Value |
|---|---|
| Architecture | DiscreteHybridVLA (PyTorch) |
| State input | observation.state 32-dim |
| Action output | action.absolute 32-dim |
| Codebook | K=512 motion primitives |
| Training loss | 0.2118 (93.1% reduction over 50 epochs) |
| Training data | public_tasks/task6911 (13,814 samples) |
| Checkpoint | checkpoint_team28_v6.pt (14.1 MB) |
| Submission | s3://airoa-icra-team-28/submission_v6/ |

## Dependencies (no CLIP, no JAX)

```
torch==2.2.0
torchvision==0.17.0
numpy pandas boto3 pyarrow
opencv-python-headless Pillow
```
