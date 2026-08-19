import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerImageProcessor

from app.vitiligo_gate import apply_vitiligo_gate


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Unified condition mask labels:
#   0 = background / normal
#   1 = vitiligo
#   2 = melasma / hyperpigmentation
#   3 = wine_stain
#   4 = forehead_wrinkle
#   5 = crow_s_feet
#   6 = nasolabial_fold
UNIFIED_LABELS = {
    "background": 0,
    "vitiligo": 1,
    "melasma": 2,
    "wine_stain": 3,
    "forehead_wrinkle": 4,
    "crow_s_feet": 5,
    "nasolabial_fold": 6,
}

CONDITIONS = [
    "vitiligo",
    "melasma",
    "wine_stain",
    "forehead_wrinkle",
    "crow_s_feet",
    "nasolabial_fold",
]

# If prediction area is smaller than this percentage of the full image,
# it is treated as noise.
MIN_AREA_PERCENT = 0.10
# Predictions covering nearly the entire face are treated as a failed class
# collapse rather than a localized condition.
MAX_AREA_PERCENT = 60.0

# Keep lower-confidence melasma pixels for multimodal confirmation. They are
# never reported as melasma unless the region analyser independently observes
# spots in more than one facial zone.
MELASMA_CANDIDATE_THRESHOLD = 0.55

_MODEL: Optional[dict] = None
_INDEPENDENT_MODELS: Optional[List[dict]] = None


def _find_project_dir() -> Path:
    current = Path(__file__).resolve()

    for parent in [current.parent] + list(current.parents):
        if (parent / "training" / "checkpoints").exists():
            return parent

    try:
        return current.parents[2]
    except IndexError:
        return current.parent


PROJECT_DIR = _find_project_dir()
SEGFORMER_ROOT = PROJECT_DIR / "training" / "checkpoints" / "SegFormer"
INDEPENDENT_MODELS_ROOT = SEGFORMER_ROOT / "models"


def _get_independent_models() -> List[dict]:
    """Load only models that passed cross-disease deployment validation."""
    global _INDEPENDENT_MODELS
    if _INDEPENDENT_MODELS is not None:
        return _INDEPENDENT_MODELS

    bundles = []
    for directory in sorted(INDEPENDENT_MODELS_ROOT.glob("*")):
        deployment_path = directory / "deployment.json"
        if not _path_has_hf_model(directory) or not deployment_path.exists():
            continue
        deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
        metrics = deployment.get("metrics", {})
        meets_quality_gate = (
            float(metrics.get("iou", 0)) >= 0.40
            and float(metrics.get("precision", 0)) >= 0.60
            and float(metrics.get("recall", 0)) >= 0.50
        )
        if not meets_quality_gate and not bool(deployment.get("enabled", False)):
            print(f"[SegFormer] Skipping unqualified model: {directory}")
            continue
        model = SegformerForSemanticSegmentation.from_pretrained(directory).to(DEVICE).eval()
        image_size = int(deployment.get("image_size", getattr(model.config, "image_size", 224)))
        processor = SegformerImageProcessor(
            do_resize=True,
            size={"height": image_size, "width": image_size},
            do_normalize=True,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
            do_reduce_labels=False,
        )
        target = _normalize_label_name(deployment["target"])
        bundles.append({
            "model": model,
            "processor": processor,
            "target": target,
            "threshold": float(deployment["threshold"]),
            "metrics": metrics,
            "path": str(directory),
        })
        print(f"[SegFormer] Active independent model: {target} ({directory})")
    _INDEPENDENT_MODELS = bundles
    return bundles


def _predict_independent(img_array: np.ndarray, bundle: dict) -> np.ndarray:
    h, w = img_array.shape[:2]
    image = Image.fromarray(img_array.astype(np.uint8)).convert("RGB")
    inputs = bundle["processor"](images=image, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.no_grad():
        logits = bundle["model"](**inputs).logits
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits.softmax(1)[0, 1].cpu().numpy()

# The joint/rebalanced multitask checkpoint (see training/TRAINING_HISTORY.md).
# A single 4-class model avoids running three independently-trained,
# per-disease detectors that can disagree on the same region.
DEFAULT_UNIFIED_MODEL_CANDIDATES = [
    "training/checkpoints/SegFormer/deprecated/unified/best",
    "training/checkpoints/SegFormer/deprecated/unified/last",
    "training/checkpoints/SegFormer/deprecated/segformer_b2_4class_port_wine_stain_finetune/best",
    "training/checkpoints/SegFormer/deprecated/segformer_b2_4class_port_wine_stain_finetune/last",
]


def _normalize_label_name(name: str) -> str:
    name = str(name).lower().strip()
    name = name.replace(" ", "_").replace("-", "_")

    aliases = {
        "background": "background",
        "bg": "background",
        "normal": "background",
        "normal_skin": "background",
        "healthy_skin": "background",
        "skin": "background",

        "vitiligo": "vitiligo",

        "melasma": "melasma",
        "melasma_like_hyperpigmentation": "melasma",
        "hyperpigmentation": "melasma",
        "black_spot": "melasma",
        "dark_spot": "melasma",
        "brown_spot": "melasma",
        "pigmentation": "melasma",

        "port_wine_stain": "wine_stain",
        "port_wine": "wine_stain",
        "portwine": "wine_stain",
        "portwine_stain": "wine_stain",
        "wine": "wine_stain",
        "wine_stain": "wine_stain",

        "forehead_wrinkle": "forehead_wrinkle",
        "forehead_wrinkles": "forehead_wrinkle",
        "wrinkle": "forehead_wrinkle",
        "wrinkles": "forehead_wrinkle",
        "crow_s_feet": "crow_s_feet",
        "crows_feet": "crow_s_feet",
        "crow_feet": "crow_s_feet",
        "nasolabial_fold": "nasolabial_fold",
        "nasolabial_folds": "nasolabial_fold",
        "smile_line": "nasolabial_fold",
        "smile_lines": "nasolabial_fold",
    }

    return aliases.get(name, name)


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)

    if p.is_absolute():
        return p

    return PROJECT_DIR / p


def _path_has_hf_model(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False

    has_config = (path / "config.json").exists()
    has_weight = (
        (path / "model.safetensors").exists()
        or (path / "pytorch_model.bin").exists()
        or any(path.glob("*.safetensors"))
        or any(path.glob("*.bin"))
    )

    return has_config and has_weight


def _find_label_mapping_file(model_dir: Path) -> Optional[Path]:
    candidates = [
        model_dir / "label_mapping.json",
        model_dir.parent / "label_mapping.json",
        model_dir.parent.parent / "label_mapping.json",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


def _load_label_mapping(model_dir: Path, model) -> Dict[int, str]:
    label_file = _find_label_mapping_file(model_dir)

    if label_file is not None:
        with open(label_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "id2label" in raw:
            raw = raw["id2label"]
        elif isinstance(raw, dict) and "label2id" in raw:
            raw = {v: k for k, v in raw["label2id"].items()}

        if isinstance(raw, dict):
            return {int(k): _normalize_label_name(v) for k, v in raw.items()}

    return {
        int(k): _normalize_label_name(v)
        for k, v in model.config.id2label.items()
    }


def _resolve_model_path() -> Path:
    env_path = os.getenv("SEGFORMER_UNIFIED_MODEL_PATH")
    candidates = ([env_path] if env_path else []) + DEFAULT_UNIFIED_MODEL_CANDIDATES

    for candidate in candidates:
        path = _resolve_path(candidate)

        if _path_has_hf_model(path):
            return path

    checked = [str(_resolve_path(c)) for c in candidates]
    raise RuntimeError(
        "No unified SegFormer checkpoint could be found. Checked: "
        f"{checked}. Set SEGFORMER_UNIFIED_MODEL_PATH to override."
    )


def _load_image_size(model_dir: Path, model) -> int:
    # train_unified.py never writes the training image_size back into
    # config.json, so config.image_size stays at the base checkpoint's
    # default (224) even though the unified model was trained at 512px.
    # The training run's own metadata file is the source of truth.
    metadata_path = model_dir / "lumiere_unified_model.json"

    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            image_size = metadata.get("hyperparameters", {}).get("image_size")

            if image_size:
                return int(image_size)
        except Exception as exc:
            print(f"[SegFormer] Could not read {metadata_path}: {exc}")

    return int(getattr(model.config, "image_size", 224))


def _resolve_local_model_path() -> Optional[Path]:
    env_path = os.getenv("SEGFORMER_LOCAL_MODEL_PATH")
    if env_path:
        candidate = _resolve_path(env_path)
    else:
        candidate = SEGFORMER_ROOT / "best_model.pt"

    if env_path and not candidate.exists():
        raise FileNotFoundError(
            f"SEGFORMER_LOCAL_MODEL_PATH does not exist: {candidate}"
        )
    return candidate if candidate.exists() else None


def _load_local_model(model_path: Path) -> tuple:
    config_path = model_path.parent / "config.json"
    if config_path.exists():
        config = SegformerConfig.from_json_file(str(config_path))
    else:
        raise FileNotFoundError(
            f"Local SegFormer config is missing: {config_path}"
        )

    model = SegformerForSemanticSegmentation(config)
    model.to(DEVICE)
    model.eval()

    state = torch.load(str(model_path), map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)

    processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": 512, "width": 512},
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        do_reduce_labels=False,
    )

    id2label = {
        int(key): _normalize_label_name(value)
        for key, value in config.id2label.items()
    }
    return model, processor, id2label


def _get_model() -> dict:
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    local_model_path = _resolve_local_model_path()
    if local_model_path is not None:
        print(f"[SegFormer] Loading local trained model: {local_model_path}")
        model, processor, id2label = _load_local_model(local_model_path)
        print(f"[SegFormer] Active model path set to: {local_model_path}")
        _MODEL = {
            "model_dir": str(local_model_path),
            "model": model,
            "processor": processor,
            "id2label": id2label,
        }
        return _MODEL

    model_dir = _resolve_model_path()
    print(f"[SegFormer] Loading unified model: {model_dir}")

    model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()

    image_size = _load_image_size(model_dir, model)

    processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": image_size, "width": image_size},
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        do_reduce_labels=False,
    )

    id2label = _load_label_mapping(model_dir, model)
    print(f"[SegFormer] labels: {id2label} (image_size={image_size})")

    _MODEL = {
        "model_dir": str(model_dir),
        "model": model,
        "processor": processor,
        "id2label": id2label,
    }
    return _MODEL


def _clean_binary_mask(binary: np.ndarray) -> np.ndarray:
    binary = binary.astype(np.uint8)

    if binary.sum() == 0:
        return binary

    h, w = binary.shape
    min_pixels = max(20, int(h * w * 0.0002))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    cleaned = np.zeros_like(binary, dtype=np.uint8)

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])

        if area >= min_pixels:
            cleaned[labels == label_idx] = 1

    return cleaned


def _predict_unified(img_array: np.ndarray) -> np.ndarray:
    bundle = _get_model()
    model = bundle["model"]
    processor = bundle["processor"]

    h, w = img_array.shape[:2]
    pil_img = Image.fromarray(img_array.astype(np.uint8)).convert("RGB")

    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        if logits.shape[1] == 1:
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (probs > 0.5).astype(np.uint8)
        else:
            logits = F.interpolate(
                logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    return pred


def _estimate_zones(binary_mask: np.ndarray) -> List[str]:
    h, w = binary_mask.shape
    ys, xs = np.where(binary_mask > 0)

    if len(xs) == 0:
        return []

    zones = set()

    if np.mean(ys < int(0.35 * h)) > 0.15:
        zones.add("forehead")

    if np.mean(xs < int(0.45 * w)) > 0.15:
        zones.add("left_cheek")

    if np.mean(xs > int(0.55 * w)) > 0.15:
        zones.add("right_cheek")

    if np.mean((xs >= int(0.35 * w)) & (xs <= int(0.65 * w))) > 0.15:
        zones.add("center_face")

    if np.mean(ys > int(0.72 * h)) > 0.15:
        zones.add("chin")

    return sorted(zones)


def get_condition_outputs(img_array: np.ndarray) -> dict:
    img_array = np.asarray(img_array)
    h, w = img_array.shape[:2]
    unified_mask = np.zeros((h, w), dtype=np.uint8)
    condition_candidates = {}

    condition_map = {
        "vitiligo": {"detected": False, "area_percent": 0, "zones": []},
        "melasma": {"detected": False, "area_percent": 0, "zones": []},
        "wine_stain": {"detected": False, "area_percent": 0, "zones": []},
        "forehead_wrinkle": {"detected": False, "area_percent": 0, "zones": []},
        "crow_s_feet": {"detected": False, "area_percent": 0, "zones": []},
        "nasolabial_fold": {"detected": False, "area_percent": 0, "zones": []},
    }

    independent_models = _get_independent_models()
    if independent_models:
        # Resolve overlaps by the strongest normalized disease probability.
        best_probability = np.zeros((h, w), dtype=np.float32)
        candidate_masks = {}
        candidate_probabilities = {}
        for bundle in independent_models:
            probability = _predict_independent(img_array, bundle)
            binary = _clean_binary_mask(
                (probability >= bundle["threshold"]).astype(np.uint8)
            )
            condition = bundle["target"]
            candidate_masks[condition] = binary
            candidate_probabilities[condition] = probability

        for condition, binary in candidate_masks.items():
            area_percent = round((int(binary.sum()) / float(h * w)) * 100, 2)
            detected = MIN_AREA_PERCENT <= area_percent <= MAX_AREA_PERCENT
            if not detected:
                continue
            probability = candidate_probabilities[condition]
            wins = (binary == 1) & (probability > best_probability)
            unified_mask[wins] = UNIFIED_LABELS[condition]
            best_probability[wins] = probability[wins]
            condition_map[condition] = {
                "detected": True,
                "area_percent": area_percent,
                "zones": _estimate_zones(binary),
            }

        # A high deployment threshold is useful for precision, but it used to
        # erase all evidence for diffuse, low-contrast melasma. Preserve that
        # evidence for the pipeline's independent Spot confirmation step.
        melasma_probability = candidate_probabilities.get("melasma")
        if (
            melasma_probability is not None
            and not condition_map["melasma"]["detected"]
        ):
            candidate_mask = _clean_binary_mask(
                (melasma_probability >= MELASMA_CANDIDATE_THRESHOLD).astype(np.uint8)
            )
            candidate_area = round(
                (int(candidate_mask.sum()) / float(h * w)) * 100, 2
            )
            if MIN_AREA_PERCENT <= candidate_area <= MAX_AREA_PERCENT:
                candidate_zones = _estimate_zones(candidate_mask)
                condition_candidates["melasma"] = {
                    "mask": candidate_mask,
                    "threshold": MELASMA_CANDIDATE_THRESHOLD,
                    "area_percent": candidate_area,
                    "zones": candidate_zones,
                }
                condition_map["melasma"]["candidate"] = {
                    "threshold": MELASMA_CANDIDATE_THRESHOLD,
                    "area_percent": candidate_area,
                    "zones": candidate_zones,
                    "max_probability": round(float(melasma_probability.max()), 4),
                }
        apply_vitiligo_gate(img_array, unified_mask, condition_map)
        return {
            "condition_mask": unified_mask,
            "condition_map": condition_map,
            "condition_candidates": condition_candidates,
        }

    # Legacy fallback is retained only for installations without promoted
    # independent models.
    bundle = _get_model()
    id2label = bundle["id2label"]
    pred = _predict_unified(img_array)

    for condition in CONDITIONS:
        target_ids = [
            label_id for label_id, label_name in id2label.items()
            if label_name == condition
        ]

        if not target_ids:
            continue

        binary = _clean_binary_mask(np.isin(pred, target_ids).astype(np.uint8))
        area_percent = round((int(binary.sum()) / float(h * w)) * 100, 2)
        detected = MIN_AREA_PERCENT <= area_percent <= MAX_AREA_PERCENT

        if detected:
            unified_mask[binary == 1] = UNIFIED_LABELS[condition]

            condition_map[condition] = {
                "detected": True,
                "area_percent": area_percent,
                "zones": _estimate_zones(binary),
            }

    apply_vitiligo_gate(img_array, unified_mask, condition_map)
    return {
        "condition_mask": unified_mask,
        "condition_map": condition_map,
        "condition_candidates": condition_candidates,
    }


def get_condition_mask(img_array: np.ndarray) -> np.ndarray:
    return get_condition_outputs(img_array)["condition_mask"]
