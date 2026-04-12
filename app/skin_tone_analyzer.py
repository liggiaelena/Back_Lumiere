
import os
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
 
# ── Resolve paths relative to this file ─────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(BASE_DIR, "checkpoints", "bisenet_best.pth")
NUM_CLASSES = 14
SKIN_LABEL  = 1       # BiSeNet label index for skin region
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
 
# ── Lazy-load model once at startup, not on every request ────────────────────
_model = None
 
def _get_model():
    global _model
    if _model is None:
        from app.models.model import BiSeNet
        net = BiSeNet(n_classes=NUM_CLASSES)
        state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        net.load_state_dict(state)
        net.eval()
        net.to(DEVICE)
        _model = net
    return _model
 
 
def analyze_skin_tone(img_rgb: np.ndarray) -> dict:
    """
    Takes the full face image as a NumPy RGB array
    (same img_rgb that pipeline.py already receives).
 
    Returns:
        {
            "mean_rgb":        [R, G, B],
            "mean_hex":        "#RRGGBB",
            "median_rgb":      [R, G, B],
            "median_hex":      "#RRGGBB",
            "num_skin_pixels": int
        }
    On failure returns all-None dict so the rest of the report is unaffected.
    """
    try:
        net = _get_model()
 
        pil_img = Image.fromarray(img_rgb)          # already RGB
        original_size = pil_img.size                # (W, H)
 
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
 
        with torch.no_grad():
            output  = net(tensor)[0]
            parsing = output.squeeze(0).argmax(0).cpu().numpy()
 
        # Resize mask back to original image dimensions
        mask = cv2.resize(
            parsing.astype(np.uint8),
            original_size,
            interpolation=cv2.INTER_NEAREST
        )
 
        # Extract only skin-labelled pixels
        img_arr     = np.array(pil_img)             # H x W x 3
        skin_pixels = img_arr[mask == SKIN_LABEL]
 
        if len(skin_pixels) == 0:
            return _empty_result()
 
        mean_rgb   = np.mean(skin_pixels,   axis=0).astype(int).tolist()
        median_rgb = np.median(skin_pixels, axis=0).astype(int).tolist()
 
        return {
            "mean_rgb":        mean_rgb,
            "mean_hex":        _to_hex(mean_rgb),
            "median_rgb":      median_rgb,
            "median_hex":      _to_hex(median_rgb),
            "num_skin_pixels": int(len(skin_pixels)),
        }
 
    except Exception as e:
        import traceback
        traceback.print_exc()
        return _empty_result()
 
 
# ── Helpers
def _to_hex(rgb: list) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)
 
def _empty_result() -> dict:
    return {
        "mean_rgb":        None,
        "mean_hex":        None,
        "median_rgb":      None,
        "median_hex":      None,
        "num_skin_pixels": 0,
    }