import os
import numpy as np
import torch
from .model import DiscreteHybridVLA, STATE_MEAN, STATE_STD, ACTION_MEAN, ACTION_STD

class HSRAdapter:
    """
    Adapter for DiscreteHybridVLA to the HSR policy server interface.
    Converts 32-dim HSR state → 8-dim internal → 8-dim action → 32-dim HSR action.
    """

    # Active state dimensions (non-zero std from norm_stats.json)
    ACTIVE_STATE_DIMS  = [0, 1, 2, 3, 4, 6, 11, 12]
    # Active action dimensions (non-zero std from norm_stats.json)
    ACTIVE_ACTION_DIMS = [0, 1, 2, 3, 4, 6, 11, 12, 13, 14, 15]

    def __init__(self, checkpoint_path=None, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model  = DiscreteHybridVLA(K=512, d=256, state_dim=8, action_dim=8).to(self.device)

        # Load checkpoint
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(__file__), "../../../checkpoint_team28_v6.pt")

        if os.path.exists(checkpoint_path):
            ckpt  = torch.load(checkpoint_path, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            miss, unex = self.model.load_state_dict(state, strict=False)
            print(f"[HSRAdapter] Loaded checkpoint: {checkpoint_path}")
            print(f"[HSRAdapter] Missing={len(miss)}, Unexpected={len(unex)}")
        else:
            print(f"[HSRAdapter] WARNING: checkpoint not found at {checkpoint_path}")
            print(f"[HSRAdapter] Running with random weights")

        self.model.eval()
        print(f"[HSRAdapter] Running on {self.device}")
        self._step = 0

    def predict(self, observation, task_index=0):
        """
        Args:
            observation: dict with "state" key -> np.array shape (32,)
                         OR np.array shape (32,)
        Returns:
            dict with "actions" key -> np.array shape (32,)
        """
        # Extract state
        if isinstance(observation, dict):
            state_32 = np.array(
                observation.get("state",
                observation.get("observation.state",
                observation.get("obs", np.zeros(32)))),
                dtype=np.float32)
        else:
            state_32 = np.array(observation, dtype=np.float32)

        # Pad/trim to 32 dims
        if len(state_32) < 32:
            state_32 = np.pad(state_32, (0, 32 - len(state_32)))
        state_32 = state_32[:32]

        # Normalise using baseline norm stats
        state_norm = (state_32 - STATE_MEAN) / STATE_STD

        # Extract active 8 dims for our model
        state_8 = state_norm[[0, 1, 2, 3, 4, 6, 11, 12]]

        # Predict 8-dim action
        with torch.no_grad():
            action_8 = self.model.predict(state_8, task_index=task_index)

        # Denormalise
        action_32      = ACTION_MEAN.copy()  # start from mean
        active         = [0, 1, 2, 3, 4, 6, 11, 12]
        for i, dim in enumerate(active):
            action_32[dim] = action_8[i] * ACTION_STD[dim] + ACTION_MEAN[dim]

        self._step += 1
        return {"actions": action_32}

    def reset(self):
        """Called at the start of each new episode."""
        self._step = 0

