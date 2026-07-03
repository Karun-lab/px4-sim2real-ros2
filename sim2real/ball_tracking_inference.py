# sim2real/iris_inference_utils.py
#!/home/karun/venvs/rl/bin/python3
"""
Inference utilities for Iris drone RL policies.
Supports both:
  - IrisBallModel  (RGB + search_active, 4ch)

Tested against SKRL >= 1.3.0  (Model.__init__ takes no positional args;
spaces are passed via class attributes).
"""

import torch
import torch.nn as nn
import gymnasium as gym

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
# =============================================================================
# YELLOW DETECTION  (ball task)
# =============================================================================

def detect_yellow(rgb: torch.Tensor, threshold: float = 0.002):
    """
    Detect yellow pixels in (N, H, W, 3) float image in [0, 1].
    Returns: visible (N,) bool, cx_norm (N,) in [-1, 1], area (N,) float
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask    = (r > 0.6) & (g > 0.5) & (b < 0.35)
    N, H, W = rgb.shape[:3]
    area    = mask.float().sum(dim=(1, 2)) / (H * W)
    visible = area > threshold

    col_idx  = torch.arange(W, device=rgb.device, dtype=torch.float32)
    col_idx  = col_idx.view(1, 1, W).expand(N, H, W)
    mask_f   = mask.float()
    cx_pixel = (mask_f * col_idx).sum(dim=(1, 2)) / mask_f.sum(dim=(1, 2)).clamp(min=1.0)
    cx_norm  = ((cx_pixel / (W - 1)) * 2.0 - 1.0) * visible.float()

    return visible, cx_norm, area


# =============================================================================
# DEPTH OPENING DETECTION  (door task)
# =============================================================================

def detect_opening(depth: torch.Tensor,
                   depth_clip: float = 12.0,
                   opening_thresh: float = 0.15,
                   far_frac: float = 0.80):
    """
    Detect a large depth opening (gap/door) in a depth image.

    depth : (N, H, W, 1) or (N, H, W) float, raw metres.
    Returns: visible (N,) bool, cx_norm (N,) in [-1, 1],
             open_frac (N,) float, depth_norm (N, H, W) float in [0, 1].
    """
    if depth.dim() == 4:
        d = depth[..., 0].clone()
    else:
        d = depth.clone()

    d = torch.nan_to_num(d, nan=depth_clip, posinf=depth_clip)
    d = d.clamp(0.0, depth_clip) / depth_clip          # normalise [0, 1]

    open_mask = d > far_frac
    N, H, W   = d.shape
    open_frac = open_mask.float().sum(dim=(1, 2)) / (H * W)
    visible   = open_frac > opening_thresh

    col_idx  = torch.arange(W, device=d.device, dtype=torch.float32).view(1, 1, W).expand(N, H, W)
    mask_f   = open_mask.float()
    cx_pixel = (mask_f * col_idx).sum(dim=(1, 2)) / mask_f.sum(dim=(1, 2)).clamp(min=1.0)
    cx_norm  = ((cx_pixel / (W - 1)) * 2.0 - 1.0) * visible.float()

    return visible, cx_norm, open_frac, d


# =============================================================================
# SHARED CNN+MLP BACKBONE
# =============================================================================

def _build_cnn(n_ch: int):
    """Returns the shared CNN body and its output size for 64×64 input."""
    cnn = nn.Sequential(
        nn.Conv2d(n_ch, 32,  kernel_size=5, stride=2),
        nn.BatchNorm2d(32),  nn.ReLU(),
        nn.Conv2d(32,  64,  kernel_size=5, stride=2),
        nn.BatchNorm2d(64),  nn.ReLU(),
        nn.Conv2d(64,  128, kernel_size=4, stride=2),
        nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, stride=2),
        nn.BatchNorm2d(256), nn.ReLU(),
        nn.Flatten(),
    )
    with torch.no_grad():
        cnn_out = cnn(torch.zeros(1, n_ch, 64, 64)).shape[1]
    return cnn, cnn_out


# =============================================================================
# SKRL MODEL BASE — compatible with SKRL >= 1.3.0
# =============================================================================
# In SKRL >= 1.3 the Model base class changed its __init__ signature.
# Spaces are injected as class-level attributes BEFORE calling __init__,
# then __init__() is called with no positional arguments.
#
# The helper _make_model_class() below creates a fresh subclass whose class
# body already contains the correct spaces, avoiding the version-dependent
# positional-arg problem entirely.

def _make_model_class(obs_space, act_space, n_ch: int, t_steps: int = 3):
    """
    Factory that returns a fully-defined Model subclass with spaces baked in.
    Works with both old and new SKRL APIs.
    """

    class _IrisModel(GaussianMixin, DeterministicMixin, Model):

        # ── Class-level space injection (new SKRL API) ─────────────────────
        observation_space = obs_space
        action_space      = act_space

        def __init__(self, device):
            # Try new API first (no positional args), fall back to old API.
            try:
                Model.__init__(self, obs_space, act_space, device)
            except TypeError:
                # SKRL >= 1.3: spaces already set as class attrs
                Model.__init__(self)
                self.device = device

            GaussianMixin.__init__(self,
                clip_actions=False, clip_log_std=True,
                min_log_std=-20.0, max_log_std=2.0, reduction="sum")
            DeterministicMixin.__init__(self, clip_actions=False)

            self.t_steps = t_steps
            self.h = self.w = 64
            self.n_ch = n_ch

            self.cnn, self.cnn_out = _build_cnn(n_ch)

            self.net = nn.Sequential(
                nn.Linear(t_steps * self.cnn_out, 512),
                nn.LayerNorm(512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
            )

            self.policy_mean = nn.Linear(256, act_space.shape[0])
            self.log_std     = nn.Parameter(torch.zeros(act_space.shape[0]))
            self.value_head  = nn.Linear(256, 1)

        def act(self, inputs, role):
            if role == "policy":
                return GaussianMixin.act(self, inputs, role)
            return DeterministicMixin.act(self, inputs, role)

        def compute(self, inputs, role=""):
            obs = inputs.get("states")
            N   = obs.shape[0]
            # Unflatten manually — no Isaac Lab dependency needed here
            obs = obs.reshape(N, self.t_steps, self.h, self.w, self.n_ch)

            feats = []
            for t in range(self.t_steps):
                frame = obs[:, t].permute(0, 3, 1, 2)   # (N, C, H, W)
                feats.append(self.cnn(frame))

            shared = self.net(torch.cat(feats, dim=1))

            if role == "policy":
                return self.policy_mean(shared), self.log_std, {}
            elif role == "value":
                return self.value_head(shared), {}
            return self.policy_mean(shared), self.log_std, {}

    return _IrisModel


# =============================================================================
# PUBLIC MODEL CLASSES  (named for clarity in checkpoints / logs)
# =============================================================================

def _make_ball_model_cls():
    obs = gym.spaces.Box(low=0., high=1., shape=(3, 64, 64, 4), dtype=float)
    act = gym.spaces.Box(low=-1., high=1., shape=(2,))
    return _make_model_class(obs, act, n_ch=4)


def _make_door_model_cls():
    obs = gym.spaces.Box(low=0., high=1., shape=(3, 64, 64, 2), dtype=float)
    act = gym.spaces.Box(low=-1., high=1., shape=(2,))
    return _make_model_class(obs, act, n_ch=2)


# =============================================================================
# LOADERS
# =============================================================================

def _load(model_cls, checkpoint_path: str, device: torch.device):
    """Internal: instantiate, load weights, return eval model."""
    model = model_cls(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # SKRL saves policy weights under "policy" key; handle raw state_dicts too
    if isinstance(ckpt, dict) and "policy" in ckpt:
        state_dict = ckpt["policy"]
    else:
        state_dict = ckpt

    # Strip "module." prefix if the model was saved with DataParallel
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_policy] WARNING — missing keys:    {missing}")
    if unexpected:
        print(f"[load_policy] WARNING — unexpected keys: {unexpected}")

    model.eval()
    return model


def load_policy(checkpoint_path: str, device: torch.device):
    """
    Load a trained BALL-TRACKING policy (.pt checkpoint).
    Returns model in eval mode, ready for act_inference().

    Usage:
        model = load_policy("best_agent.pt", torch.device("cpu"))
        action = model.act_inference(obs_history)
    """
    return _load(_make_ball_model_cls(), checkpoint_path, device)


def load_door_policy(checkpoint_path: str, device: torch.device):
    """
    Load a trained DOOR-FINDING policy (.pt checkpoint).
    Returns model in eval mode, ready for act_inference().
    """
    return _load(_make_door_model_cls(), checkpoint_path, device)


# =============================================================================
# INFERENCE HELPER  (shared by both ROS nodes)
# =============================================================================

def build_obs_history(new_frame: torch.Tensor,
                      history: torch.Tensor | None,
                      history_len: int = 3) -> torch.Tensor:
    """
    Append new_frame (1, H, W, C) to rolling history (1, T, H, W, C).
    Initialises with frame repetition on first call.
    Returns updated history tensor.
    """
    frame4d = new_frame.unsqueeze(1)   # (1, 1, H, W, C)
    if history is None:
        return frame4d.repeat(1, history_len, 1, 1, 1)
    return torch.cat([history[:, 1:], frame4d], dim=1)


def run_policy(model, history: torch.Tensor) -> list[float]:
    """
    Run a deterministic forward pass and return [vx, yaw_rate] as plain floats.
    history: (1, T, H, W, C)
    """
    # Flatten to (1, T*H*W*C) — what compute() expects as inputs["states"]
    flat = history.reshape(1, -1)
    with torch.no_grad():
        # act_inference uses the deterministic (mean) action — no sampling
        action, _, _ = model.act({"states": flat}, role="policy")
        
        #action, _, _ = model.act({"states": flat})
    return [float(action[0, 0]), float(action[0, 1])]