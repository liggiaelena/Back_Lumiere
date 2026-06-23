import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Unified condition mask labels:
#   0 = background / normal
#   1 = vitiligo
#   2 = melasma
#   3 = wine_stain
UNIFIED_LABELS = {
    "background": 0,
    "vitiligo": 1,
    "melasma": 2,
    "wine_stain": 3,
}

CONDITIONS = ["vitiligo", "melasma", "wine_stain"]

# If prediction area is smaller than this percentage of the full image,
# it is treated as noise.
MIN_AREA_PERCENT = 0.10

_MODELS = {}


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


DEFAULT_MODEL_SOURCES = {
    "vitiligo": [
        {
            "path": "training/checkpoints/SegFormer/segformer_b2_4class_vitiligo_finetune/vitiligo_finetune/best",
            "threshold": None,
        },
        {
            "path": "training/checkpoints/SegFormer/segformer_b2_4class_vitiligo_finetune/vitiligo_finetune",
            "threshold": None,
        },
        {
            "path": "training/checkpoints/SegFormer/segformer_b2_4class_vitiligo_finetune/best",
            "threshold": None,
        },
        {
            "path": "training/checkpoints/SegFormer/segformer_b2_4class_vitiligo_finetune",
            "threshold": None,
        },
    ],
    "melasma": [
        {
            "path": "training/checkpoints/SegFormer/segformer_b2_melasma_colab_export/best",
            "threshold": 0.30,
        },
        {
            "path": "training/checkpoints/SegFormer/segformer_b2_melasma_colab_export",
            "threshold": 0.30,
        },
        {
            "path": "training/checkpoints/SegFormer/segformer_melasma_colab_export/best",
            "threshold": 0.60,
        },
        {
            "path": "training/checkpoints/SegFormer/segformer_melasma_colab_export",
            "threshold": 0.60,
        },
    ],
    "wine_stain": [
        {
            "path": "training/checkpoints/SegFormer/segformer_b2_4class_port_wine_stain_finetune/best",
            "threshold": None,
        },
        {
            "path": "training/checkpoints/SegFormer/segformer_b2_4class_port_wine_stain_finetune",
            "threshold": None,
        },
    ],
}


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

        "port_wine_stain": "wine_stain",
        "port_wine": "wine_stain",
        "portwine": "wine_stain",
        "portwine_stain": "wine_stain",
        "wine": "wine_stain",
        "wine_stain": "wine_stain",
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


def _load_label_mapping(model_dir: Path, model, fallback_condition: str) -> Dict[int, str]:
    label_file = _find_label_mapping_file(model_dir)

    if label_file is not None:
        with open(label_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "id2label" in raw:
            raw = raw["id2label"]
        elif isinstance(raw, dict) and "label2id" in raw:
            raw = raw["label2id"]

        if isinstance(raw, dict):
            if all(str(k).isdigit() for k in raw.keys()):
                return {
                    int(k): _normalize_label_name(v)
                    for k, v in raw.items()
                }

            return {
                int(v): _normalize_label_name(k)
                for k, v in raw.items()
            }

    num_labels = int(getattr(model.config, "num_labels", 2))

    if hasattr(model.config, "id2label") and model.config.id2label:
        id2label = {
            int(k): _normalize_label_name(v)
            for k, v in model.config.id2label.items()
        }

        values = set(id2label.values())

        if num_labels == 2 and not (values & set(CONDITIONS)):
            return {
                0: "background",
                1: fallback_condition,
            }

        return id2label

    if num_labels == 1:
        return {
            0: fallback_condition,
        }

    if num_labels == 2:
        return {
            0: "background",
            1: fallback_condition,
        }

    return {
        0: "background",
        1: fallback_condition,
    }


def _read_env_sources() -> Dict[str, List[dict]]:
    raw = os.getenv("SEGFORMER_SOURCES_JSON")

    if not raw:
        return {}

    try:
        data = json.loads(raw)
    except Exception as exc:
        print(f"[SegFormer] Invalid SEGFORMER_SOURCES_JSON: {exc}")
        return {}

    out = {}

    for condition, sources in data.items():
        condition = _normalize_label_name(condition)

        if condition not in CONDITIONS:
            continue

        if isinstance(sources, str):
            out[condition] = [{"path": sources, "threshold": None}]
        elif isinstance(sources, list):
            normalized_sources = []

            for item in sources:
                if isinstance(item, str):
                    normalized_sources.append({"path": item, "threshold": None})
                elif isinstance(item, dict) and "path" in item:
                    normalized_sources.append({
                        "path": item["path"],
                        "threshold": item.get("threshold"),
                    })

            out[condition] = normalized_sources

    return out


def _auto_find_checkpoint(condition: str) -> Optional[dict]:
    if not SEGFORMER_ROOT.exists():
        return None

    keywords = {
        "vitiligo": ["vitiligo"],
        "melasma": ["melasma"],
        "wine_stain": ["wine", "port"],
    }[condition]

    candidates = []

    for folder in SEGFORMER_ROOT.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name.lower()

        if not any(keyword in folder_name for keyword in keywords):
            continue

        possible_model_dirs = [
            folder / "best",
            folder,
            folder / "last",
            folder / "vitiligo_finetune" / "best",
            folder / "vitiligo_finetune",
        ]

        for p in possible_model_dirs:
            if _path_has_hf_model(p):
                threshold = None

                if condition == "melasma":
                    if "b2_melasma_colab_export" in str(p).lower():
                        threshold = 0.30
                    elif "melasma_colab_export" in str(p).lower():
                        threshold = 0.60

                candidates.append({"path": p, "threshold": threshold})

    candidates = sorted(
        candidates,
        key=lambda item: (
            0 if Path(item["path"]).name == "best" else 1,
            len(str(item["path"])),
        ),
    )

    return candidates[0] if candidates else None


def _select_source_for_condition(condition: str) -> Optional[dict]:
    env_sources = _read_env_sources()
    sources = env_sources.get(condition, []) + DEFAULT_MODEL_SOURCES.get(condition, [])

    for source in sources:
        p = _resolve_path(source["path"])

        if _path_has_hf_model(p):
            return {
                "path": p,
                "threshold": source.get("threshold"),
            }

    return _auto_find_checkpoint(condition)


def _load_one_model(condition: str, source: dict) -> dict:
    model_dir = Path(source["path"])
    threshold = source.get("threshold")

    print(f"[SegFormer] Loading {condition}: {model_dir}")

    model = SegformerForSemanticSegmentation.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()

    image_size = getattr(model.config, "image_size", 224)

    processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": image_size, "width": image_size},
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        do_reduce_labels=False,
    )

    id2label = _load_label_mapping(
        model_dir=model_dir,
        model=model,
        fallback_condition=condition,
    )

    print(f"[SegFormer] {condition} labels: {id2label}")

    return {
        "condition": condition,
        "model_dir": str(model_dir),
        "threshold": threshold,
        "model": model,
        "processor": processor,
        "id2label": id2label,
    }


def _get_models() -> dict:
    global _MODELS

    if _MODELS:
        return _MODELS

    loaded = {}

    for condition in CONDITIONS:
        source = _select_source_for_condition(condition)

        if source is None:
            print(f"[SegFormer] No checkpoint found for {condition}.")
            continue

        try:
            loaded[condition] = _load_one_model(condition, source)
        except Exception as exc:
            print(f"[SegFormer] Failed to load {condition}: {exc}")

    if not loaded:
        raise RuntimeError(
            "No SegFormer checkpoints could be loaded. "
            "Check training/checkpoints/SegFormer or set SEGFORMER_SOURCES_JSON."
        )

    _MODELS = loaded
    return _MODELS


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


def _target_ids_for_condition(bundle: dict, num_output_classes: int) -> List[int]:
    condition = bundle["condition"]
    id2label = bundle["id2label"]

    target_ids = []

    for label_id, label_name in id2label.items():
        if _normalize_label_name(label_name) == condition:
            target_ids.append(int(label_id))

    if target_ids:
        return target_ids

    if num_output_classes == 1:
        return [0]

    if num_output_classes == 2:
        return [1]

    for label_id, label_name in id2label.items():
        if _normalize_label_name(label_name) != "background":
            target_ids.append(int(label_id))

    return target_ids


def _predict_condition_binary(img_array: np.ndarray, bundle: dict) -> np.ndarray:
    model = bundle["model"]
    processor = bundle["processor"]
    threshold = bundle["threshold"]

    h, w = img_array.shape[:2]

    pil_img = Image.fromarray(img_array.astype(np.uint8)).convert("RGB")

    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        num_output_classes = int(logits.shape[1])
        target_ids = _target_ids_for_condition(bundle, num_output_classes)

        if threshold is not None and len(target_ids) == 1:
            target_id = target_ids[0]

            if num_output_classes == 1:
                prob = torch.sigmoid(logits[:, 0, :, :])[0]
            else:
                prob = torch.softmax(logits, dim=1)[0, target_id, :, :]

            binary = (prob.cpu().numpy() >= float(threshold)).astype(np.uint8)
        else:
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            binary = np.isin(pred, target_ids).astype(np.uint8)

    return _clean_binary_mask(binary)


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
    models = _get_models()

    unified_mask = np.zeros((h, w), dtype=np.uint8)

    condition_map = {
        "vitiligo": {"detected": False, "area_percent": 0, "zones": []},
        "melasma": {"detected": False, "area_percent": 0, "zones": []},
        "wine_stain": {"detected": False, "area_percent": 0, "zones": []},
    }

    merge_order = ["melasma", "wine_stain", "vitiligo"]

    for condition in merge_order:
        if condition not in models:
            continue

        binary = _predict_condition_binary(img_array, models[condition])

        area_percent = round((int(binary.sum()) / float(h * w)) * 100, 2)
        detected = area_percent >= MIN_AREA_PERCENT

        if detected:
            unified_mask[binary == 1] = UNIFIED_LABELS[condition]

            condition_map[condition] = {
                "detected": True,
                "area_percent": area_percent,
                "zones": _estimate_zones(binary),
            }

    return {
        "condition_mask": unified_mask,
        "condition_map": condition_map,
    }


def get_condition_mask(img_array: np.ndarray) -> np.ndarray:
    return get_condition_outputs(img_array)["condition_mask"]
