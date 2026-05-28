"""
app.py — Production-ready Streamlit application for real-time SIBI alphabet recognition.

SIBI (Sistem Isyarat Bahasa Indonesia) — Static alphabet A–Y (no J, no Z).

Pipeline:
    Webcam frame
        → MediaPipe Hand Landmarker (detect hand landmarks)
        → Bounding-box crop with padding
        → Resize(128, 128) + ImageNet Normalization
        → SIBINet CNN inference
        → Display predicted letter + confidence

Hardware target: Intel 12th-Gen i5-HX · NVIDIA RTX 2050 (4 GB VRAM)

Directory layout expected:
    app.py
    outputs/
        sibi_cnn_best.pth
        class_to_idx.json
    model_handlandmark/
        hand_landmarker.task
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from torchvision import transforms

# 1.  PAGE CONFIG
st.set_page_config(
    page_title="SIBI's Sign Language Recognition",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2.  GLOBAL PATHS  — edit here if your layout differs
MODEL_WEIGHTS_PATH:  Path = Path("outputs/sibi_cnn_best.pth")
CLASS_MAP_PATH:      Path = Path("outputs/class_to_idx.json")
HAND_LANDMARKER_PATH: Path = Path("model_handlandmark/hand_landmarker.task")

# 3.  MODEL ARCHITECTURE  
class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise-separable convolution block.

    Replaces a standard 3×3 conv with:
      - a depthwise 3×3  (one filter per input channel)
      - a pointwise 1×1  (mixes channels)

    This cuts FLOPs and parameter count by ~8–9× vs a plain Conv2d.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        # Depthwise: groups=in_ch means each channel gets its own 3×3 kernel
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride,
            padding=1, groups=in_ch, bias=False,
        )
        # Pointwise: 1×1 conv that projects to the desired number of output channels
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class ResLiteBlock(nn.Module):
    """
    Lightweight residual block built from two DepthwiseSeparableConv layers.

    A projection shortcut (1×1 Conv + BN) is inserted whenever:
      - stride > 1  (spatial downsampling changes the feature-map size), or
      - in_ch != out_ch  (channel width changes).
    Otherwise a plain nn.Identity() shortcut is used (zero cost).
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch, stride=stride)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch, stride=1)

        # Shortcut path: match spatial dims and channel width if needed
        if stride != 1 or in_ch != out_ch:
            self.skip: nn.Module = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection: main path + shortcut, then activation
        return F.relu(self.conv2(self.conv1(x)) + self.skip(x), inplace=True)


class SIBINet(nn.Module):
    """
    Lightweight ResNet-style CNN for SIBI alphabet classification.

    Architecture
    ─────────────────────────────────────────────────
    Stem  : Conv 3→32, BN, ReLU, MaxPool   (128→64)
    Stage1: 2× ResLiteBlock  32→64          (64→32 via stride-2)
    Stage2: 2× ResLiteBlock  64→128         (32→16 via stride-2)
    Stage3: 2× ResLiteBlock 128→256         (16→8  via stride-2)
    Stage4: 2× ResLiteBlock 256→512         (8→4   via stride-2)
    Head  : AdaptiveAvgPool → Dropout → Linear(512, num_classes)
    ─────────────────────────────────────────────────
    Total params: ~2.1 M  |  VRAM (FP16, batch 1): < 100 MB

    Parameters
    ----------
    num_classes : Number of output logits (24 for SIBI A–Y, excluding J/Z)
    dropout     : Dropout probability before the final linear layer
    """

    def __init__(self, num_classes: int = 24, dropout: float = 0.4) -> None:
        super().__init__()

        # Stem block: initial feature extraction + first spatial downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 128 → 64
        )

        # Four progressive residual stages, each doubling the channels
        # and halving the spatial resolution via stride-2 in the first block
        self.stage1 = self._make_stage(32,  64,  n_blocks=2, stride=2)  # 64 → 32
        self.stage2 = self._make_stage(64,  128, n_blocks=2, stride=2)  # 32 → 16
        self.stage3 = self._make_stage(128, 256, n_blocks=2, stride=2)  # 16 →  8
        self.stage4 = self._make_stage(256, 512, n_blocks=2, stride=2)  #  8 →  4

        # Global average pooling collapses spatial dims → (B, 512)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(512, num_classes)

        self._init_weights()

    # helpers 

    @staticmethod
    def _make_stage(
        in_ch: int,
        out_ch: int,
        n_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Stack n_blocks ResLiteBlocks; the first block applies 'stride'."""
        layers = [ResLiteBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(ResLiteBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Kaiming-normal init for Conv2d; constant init for BN layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # forward 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W) normalised image tensor

        Returns
        -------
        logits : (B, num_classes) raw scores (NOT softmax)
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).flatten(1)   # (B, 512)
        x = self.dropout(x)
        return self.fc(x)             # (B, num_classes)



# 4.  PREPROCESSING  (mirrors notebook's val_transforms exactly)

# ImageNet statistics used during training
_IMG_SIZE: int         = 128
_MEAN: Tuple           = (0.485, 0.456, 0.406)
_STD: Tuple            = (0.229, 0.224, 0.225)

# This transform must match val_transforms in the training notebook:
#   Resize(128, 128) → ToTensor() → Normalize(ImageNet mean/std)
_INFERENCE_TRANSFORM: transforms.Compose = transforms.Compose([
    transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


def preprocess_hand(crop_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert a BGR crop (from OpenCV) into a normalised model-ready tensor.

    Steps
    -----
    1. BGR → RGB  (OpenCV uses BGR; PIL and the model expect RGB)
    2. numpy array → PIL Image
    3. Apply inference transforms: Resize(128,128) + ToTensor + Normalize

    Parameters
    ----------
    crop_bgr : (H, W, 3) uint8 numpy array in BGR colour order

    Returns
    -------
    tensor : (1, 3, 128, 128) float32 tensor ready for model.forward()
    """
    # Convert colour space: OpenCV BGR → PIL RGB
    rgb_image: Image.Image = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

    # Apply the same transforms used at validation/test time
    tensor: torch.Tensor = _INFERENCE_TRANSFORM(rgb_image)   # (3, 128, 128)

    # Add batch dimension: (3, 128, 128) → (1, 3, 128, 128)
    return tensor.unsqueeze(0)



# 4.  MODEL LOADING  (cached so it only loads once per Streamlit session)
@st.cache_resource(show_spinner="Loading SIBINet model…")
def load_model(
    weights_path: Path,
    class_map_path: Path,
) -> Tuple[nn.Module, Dict[int, str], torch.device]:
    """
    Load SIBINet weights and the class-index mapping from disk.

    The function is decorated with @st.cache_resource so that the
    expensive checkpoint load only happens once per Streamlit process,
    even across page refreshes.

    Parameters
    ----------
    weights_path    : Path to 'sibi_cnn_best.pth'
    class_map_path  : Path to 'class_to_idx.json'

    Returns
    -------
    model       : SIBINet in eval mode, moved to the best available device
    idx_to_class: mapping from integer class index → letter string
    device      : the torch.device the model lives on
    """
    # Device selection: prefer CUDA, fall back to CPU gracefully 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the class-index JSON produced during training
    with open(class_map_path, "r") as f:
        class_to_idx: Dict[str, int] = json.load(f)
    idx_to_class: Dict[int, str] = {v: k for k, v in class_to_idx.items()}
    num_classes: int = len(class_to_idx)

    # Instantiate model with the same hyperparameters used in training 
    model = SIBINet(num_classes=num_classes, dropout=0.4)

    # Load checkpoint; map_location ensures CPU loading works too 
    checkpoint = torch.load(weights_path, map_location=device)

    # The checkpoint stores the state dict under the key "model_state"
    # (see the training loop's torch.save() call)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)

    # Move to device and switch to inference mode 
    model = model.to(device)
    model.eval()

    # torch.compile omitted — requires Triton (Linux-only, unavailable on Windows).

    return model, idx_to_class, device


# 6.  HAND LANDMARKER  (cached for the same reason as the model)
@st.cache_resource(show_spinner="Loading MediaPipe Hand Landmarker…")
def load_hand_landmarker(task_path: Path) -> mp_vision.HandLandmarker:
    """
    Initialise the MediaPipe HandLandmarker in IMAGE mode.

    Parameters
    ----------
    task_path : Path to 'hand_landmarker.task'

    Returns
    -------
    A ready-to-use HandLandmarker instance (single-image, synchronous mode).
    """
    base_options = mp_python.BaseOptions(model_asset_path=str(task_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options    = base_options,
        running_mode    = mp_vision.RunningMode.IMAGE,  # synchronous, single-frame
        num_hands       = 1,    # only track the primary hand for SIBI
        min_hand_detection_confidence  = 0.5,
        min_hand_presence_confidence   = 0.5,
        min_tracking_confidence        = 0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# 7.  BOUNDING-BOX CROP  (dynamic crop from landmark coordinates)
_PADDING_FRACTION: float = 0.25  # 25 % padding on each side for better context

# MediaPipe hand bone connections (pairs of landmark indices)
_HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
)
_FINGERTIPS: Tuple[int, ...] = (4, 8, 12, 16, 20)


def get_hand_crop(
    frame_bgr: np.ndarray,
    landmarker: "HandLandmarker",
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int,int,int,int]], Optional[list]]:
    """
    Detect the hand, return a square padded crop, the bounding box, and
    the 21 landmark pixel coords for overlay drawing.

    Returns
    -------
    (crop_bgr, bbox, landmarks_px)
        crop_bgr     — square BGR crop, or None if no hand found
        bbox         — (x1, y1, x2, y2) padded box in pixel coords, or None
        landmarks_px — list of (px, py) for all 21 landmarks, or None

    Square crop prevents distortion when Resize(128,128) is applied —
    the model was trained on square images.
    """
    h, w = frame_bgr.shape[:2]

    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result    = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None, None, None

    landmarks    = result_lm = result.hand_landmarks[0]
    xs           = [lm.x * w for lm in result_lm]
    ys           = [lm.y * h for lm in result_lm]
    landmarks_px = [(int(x), int(y)) for x, y in zip(xs, ys)]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    pad_x = int((x_max - x_min) * _PADDING_FRACTION)
    pad_y = int((y_max - y_min) * _PADDING_FRACTION)

    rx1, ry1 = int(x_min) - pad_x, int(y_min) - pad_y
    rx2, ry2 = int(x_max) + pad_x, int(y_max) + pad_y

    # Force square: expand the shorter axis symmetrically
    cw, ch = rx2 - rx1, ry2 - ry1
    if cw < ch:
        d = ch - cw; rx1 -= d // 2; rx2 += d - d // 2
    elif ch < cw:
        d = cw - ch; ry1 -= d // 2; ry2 += d - d // 2

    x1 = max(0, rx1); y1 = max(0, ry1)
    x2 = min(w, rx2); y2 = min(h, ry2)

    if x2 <= x1 or y2 <= y1:
        return None, None, None

    return frame_bgr[y1:y2, x1:x2], (x1, y1, x2, y2), landmarks_px


# 8.  INFERENCE  (single frame → predicted letter + confidence)
@torch.inference_mode()   # leaner than torch.no_grad(); disables autograd entirely
def run_inference(
    crop_bgr: np.ndarray,
    model: nn.Module,
    idx_to_class: Dict[int, str],
    device: torch.device,
) -> Tuple[str, float]:
    """
    Run a single forward pass and return the top-1 prediction.

    Using torch.inference_mode() (instead of torch.no_grad()) prevents
    gradient tracking and view tracking, giving a small speed boost —
    ideal for a real-time inference loop.

    Parameters
    ----------
    crop_bgr     : (H, W, 3) uint8 BGR crop of the detected hand
    model        : SIBINet in eval mode
    idx_to_class : Mapping from class index → letter string
    device       : torch.device the model lives on

    Returns
    -------
    letter     : Predicted SIBI letter (e.g. "A", "B", …)
    confidence : Softmax probability of the top prediction (0–1)
    """
    # Preprocess: BGR crop → (1, 3, 128, 128) normalised tensor
    tensor: torch.Tensor = preprocess_hand(crop_bgr).to(device)

    # Optional: use AMP FP16 for RTX 2050 speed boost
    # autocast is a no-op on CPU, so this is always safe to wrap
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=(device.type == "cuda")):
        logits: torch.Tensor = model(tensor)  # (1, num_classes)

    # Convert raw logits → probability distribution
    probs: torch.Tensor = torch.softmax(logits.float(), dim=1)  # back to fp32

    # Extract top-1 prediction
    confidence_val, class_idx_tensor = probs.max(dim=1)
    class_idx:  int   = class_idx_tensor.item()   # type: ignore[assignment]
    confidence: float = confidence_val.item()      # type: ignore[assignment]

    letter: str = idx_to_class.get(class_idx, "?")
    return letter, confidence


# 9.  STREAMLIT UI
def draw_overlay(
    frame_bgr: np.ndarray,
    letter: Optional[str],
    confidence: Optional[float],
    inference_ms: Optional[float],
    hand_detected: bool,
    bbox: Optional[Tuple[int,int,int,int]] = None,
    landmarks_px: Optional[list] = None,
) -> np.ndarray:
    """Draw bounding box, skeleton, prediction badge, and status onto frame."""
    h, w = frame_bgr.shape[:2]

    accent = (0, 220, 80) if (hand_detected and confidence is not None and confidence >= 0.75)              else (0, 200, 230) if hand_detected else (0, 100, 255)

    # ── Bounding box ─────────────────────────────────────────────────────────
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        bw = x2 - x1

        # Semi-transparent fill
        overlay_r = frame_bgr.copy()
        cv2.rectangle(overlay_r, (x1, y1), (x2, y2), accent, -1)
        cv2.addWeighted(overlay_r, 0.08, frame_bgr, 0.92, 0, frame_bgr)

        # Border
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), accent, 2, cv2.LINE_AA)

        # Corner ticks
        tick = max(12, bw // 8)
        for cx, cy in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
            dx = tick if cx == x1 else -tick
            dy = tick if cy == y1 else -tick
            cv2.line(frame_bgr, (cx,cy), (cx+dx,cy), accent, 3, cv2.LINE_AA)
            cv2.line(frame_bgr, (cx,cy), (cx,cy+dy), accent, 3, cv2.LINE_AA)

        # Prediction badge pinned above the box
        if letter is not None and confidence is not None:
            label = f" {letter}  {confidence:.0%} "
            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            by1 = max(0, y1 - lh - bl - 8)
            by2 = max(lh + bl + 4, y1)
            cv2.rectangle(frame_bgr, (x1, by1), (x1 + lw, by2), accent, -1)
            cv2.putText(frame_bgr, label, (x1, by2 - bl - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20,20,20), 2, cv2.LINE_AA)

    # ── Skeleton ─────────────────────────────────────────────────────────────
    if landmarks_px is not None:
        for a, b in _HAND_CONNECTIONS:
            if a < len(landmarks_px) and b < len(landmarks_px):
                cv2.line(frame_bgr, landmarks_px[a], landmarks_px[b],
                         (220,220,220), 1, cv2.LINE_AA)
        for idx, (px, py) in enumerate(landmarks_px):
            r     = 5 if idx in _FINGERTIPS else 3
            color = (255,255,255) if idx in _FINGERTIPS else (180,180,180)
            cv2.circle(frame_bgr, (px,py), r, color, -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (px,py), r, (60,60,60), 1, cv2.LINE_AA)

    # ── Status text ───────────────────────────────────────────────────────────
    status_text  = "Hand detected" if hand_detected else "No hand detected"
    status_color = (0,220,80) if hand_detected else (0,100,255)
    cv2.putText(frame_bgr, status_text, (12,36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2, cv2.LINE_AA)

    # ── Confidence bar ────────────────────────────────────────────────────────
    if confidence is not None:
        bar_x1, bar_y = 12, h - 20
        bar_w = int((w - 24) * confidence)
        bar_c = (0,220,80) if confidence >= 0.75 else (0,200,230)
        cv2.rectangle(frame_bgr, (bar_x1, bar_y-16), (bar_x1+bar_w, bar_y), bar_c, -1)
        cv2.rectangle(frame_bgr, (bar_x1, bar_y-16), (bar_x1+w-24, bar_y), (180,180,180), 1)
        cv2.putText(frame_bgr, f"Conf: {confidence:.1%}", (bar_x1, bar_y-22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1, cv2.LINE_AA)

    # ── Latency ───────────────────────────────────────────────────────────────
    if inference_ms is not None:
        lat = f"{inference_ms:.1f} ms"
        sz, _ = cv2.getTextSize(lat, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(frame_bgr, lat, (w - sz[0] - 12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1, cv2.LINE_AA)

    return frame_bgr


def main() -> None:
    """Main Streamlit application entry point."""

    # CSS overrides for a clean dark look 
    st.markdown("""
    <style>
        /* Reduce default top padding */
        .block-container { padding-top: 1.5rem; }

        /* Prediction badge */
        .pred-badge {
            font-size: 5rem;
            font-weight: 800;
            text-align: center;
            padding: 0.2em 0.5em;
            border-radius: 0.4em;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: #e0e0e0;
            letter-spacing: 0.05em;
            border: 2px solid #3a3a5c;
        }
        /* Confidence row */
        .conf-row {
            font-size: 1.1rem;
            color: #aaaaaa;
            text-align: center;
            margin-top: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Page title
    st.title("SIBI's Sign Language Recognition")
    st.caption(
        "Real-time recognition of Indonesian Sign Language (SIBI) static alphabet A–Y. "
        "Show one hand clearly to the camera."
    )

    # Sidebar: settings and info
    with st.sidebar:
        st.header("⚙️ Settings")

        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.0, max_value=1.0, value=0.50, step=0.05,
            help="Predictions below this threshold are shown as '?'",
        )

        show_crop = st.checkbox(
            "Show hand crop", value=False,
            help="Display the pre-processed 128×128 hand crop used for inference.",
        )

        st.divider()
        st.subheader("📋 Model info")
        st.markdown("""
        | Parameter | Value |
        |---|---|
        | Architecture | SIBINet (ResNet-Lite) |
        | Input size | 128 × 128 |
        | Parameters | ~2.1 M |
        | Classes | 24 (A–Y, no J/Z) |
        | Normalization | ImageNet mean/std |
        """)

        st.divider()
        st.caption("💡 Tip: ensure good lighting and keep your hand centred.")

    # Load model and hand landmarker (cached) 
    if not MODEL_WEIGHTS_PATH.exists():
        st.error(
            f"Model weights not found at **{MODEL_WEIGHTS_PATH}**. "
            "Please ensure 'outputs/sibi_cnn_best.pth' exists.",
            icon="🚫",
        )
        st.stop()

    if not CLASS_MAP_PATH.exists():
        st.error(
            f"Class map not found at **{CLASS_MAP_PATH}**. "
            "Please ensure 'outputs/class_to_idx.json' exists.",
            icon="🚫",
        )
        st.stop()

    if not HAND_LANDMARKER_PATH.exists():
        st.error(
            f"Hand Landmarker task not found at **{HAND_LANDMARKER_PATH}**. "
            "Please ensure 'model_handlandmark/hand_landmarker.task' exists.",
            icon="🚫",
        )
        st.stop()

    model, idx_to_class, device = load_model(MODEL_WEIGHTS_PATH, CLASS_MAP_PATH)
    landmarker                  = load_hand_landmarker(HAND_LANDMARKER_PATH)

    # Main layout: two columns (camera | live prediction)
    col_cam, col_pred = st.columns([11, 9], gap="medium")

    with col_cam:
        st.subheader("📷 Live Camera Feed")
        # Streamlit's built-in webcam component — delivers JPEG frames
        camera_placeholder = st.empty()

    with col_pred:
        st.subheader("🔤 Prediction")
        pred_letter_placeholder    = st.empty()
        pred_conf_placeholder      = st.empty()
        pred_latency_placeholder   = st.empty()

        if show_crop:
            st.subheader("✂️ Hand Crop")
            crop_placeholder = st.empty()

    # Webcam capture loop using streamlit-webrtc alternative 
    # We use st.camera_input for the simplest zero-dependency approach.
    # For continuous streaming, we use a button-gated snapshot approach
    # compatible with all deployment targets (local + Streamlit Cloud).

    run_stream = st.toggle("▶ Start Recognition", value=False)

    if not run_stream:
        camera_placeholder.info("Toggle **Start Recognition** above to begin.")
        pred_letter_placeholder.markdown(
            '<div class="pred-badge">—</div>', unsafe_allow_html=True,
        )
        return

    # OpenCV camera loop 
    # Open the default webcam (index 0). Change to 1/2/… for external cameras.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error(
            "Could not open webcam (index 0). "
            "Try changing VideoCapture index in app.py, or check camera permissions.",
            icon="📷",
        )
        return

    # Set reasonable resolution — higher resolution increases latency
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS,          30)

    # 20-fps display cap — st.image() rerenders fully each call;
    # drain camera buffer every iteration but only display at target rate.
    _FRAME_INTERVAL: float = 1.0 / 20.0
    _last_frame_time: float = 0.0

    stop_button = st.button("⏹ Stop", type="secondary")

    try:
        while run_stream and not stop_button:
            # Capture a single frame 
            ret, frame_bgr = cap.read()
            if not ret:
                st.warning("Failed to read from webcam. Retrying…")
                time.sleep(0.05)
                continue
            now = time.perf_counter()
            if now - _last_frame_time < _FRAME_INTERVAL:
                continue
            _last_frame_time = now

            # Mirror the frame horizontally for a natural "mirror" view
            frame_bgr = cv2.flip(frame_bgr, 1)

            # Hand detection: crop + bbox + landmark pixel coords
            crop_bgr, bbox, landmarks_px = get_hand_crop(frame_bgr, landmarker)
            hand_detected: bool          = crop_bgr is not None

            letter:        Optional[str]   = None
            confidence:    Optional[float] = None
            inference_ms:  Optional[float] = None

            if hand_detected and crop_bgr is not None:
                # Inference 
                t_start = time.perf_counter()
                letter, confidence = run_inference(crop_bgr, model, idx_to_class, device)
                inference_ms = (time.perf_counter() - t_start) * 1_000  # → ms

                # Apply confidence threshold: show "?" if below threshold
                if confidence < confidence_threshold:
                    letter = "?"

                # Update prediction panel 
                pred_letter_placeholder.markdown(
                    f'<div class="pred-badge">{letter}</div>',
                    unsafe_allow_html=True,
                )
                conf_color = "green" if confidence >= 0.75 else "orange"
                pred_conf_placeholder.markdown(
                    f'<div class="conf-row">'
                    f'Confidence: <span style="color:{conf_color};font-weight:600;">'
                    f'{confidence:.1%}</span></div>',
                    unsafe_allow_html=True,
                )
                pred_latency_placeholder.caption(f"⚡ Inference: {inference_ms:.1f} ms")

                # Optional crop display 
                if show_crop and crop_bgr is not None:
                    crop_rgb = cv2.cvtColor(
                        cv2.resize(crop_bgr, (_IMG_SIZE, _IMG_SIZE)),
                        cv2.COLOR_BGR2RGB,
                    )
                    crop_placeholder.image(crop_rgb, caption="128×128 crop", width=128)

            else:
                # No hand in frame — reset prediction display
                pred_letter_placeholder.markdown(
                    '<div class="pred-badge">—</div>', unsafe_allow_html=True,
                )
                pred_conf_placeholder.markdown(
                    '<div class="conf-row">No hand detected</div>',
                    unsafe_allow_html=True,
                )
                pred_latency_placeholder.empty()

            # Draw overlay: bbox + skeleton + labels
            annotated_frame = draw_overlay(
                frame_bgr, letter, confidence, inference_ms, hand_detected,
                bbox=bbox, landmarks_px=landmarks_px,
            )

            # Display annotated frame (convert BGR → RGB for Streamlit) 
            camera_placeholder.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )

    finally:
        # Always release the webcam, even if an exception occurs
        cap.release()



# 10.  ENTRY POINT
if __name__ == "__main__":
    main()
