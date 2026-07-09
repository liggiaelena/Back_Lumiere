import json
from pathlib import Path
from datetime import datetime
import base64

import asyncio

import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from app.image_utils import preprocess
from app.mediapipe_utils import get_landmarks, extract_region_crops
from app.vision import analyze_region
from app.color_utils import build_final_report
from app.skin_tone_analyzer import analyze_skin_tone


try:
    from app.segmentation import get_condition_outputs
except Exception as exc:
    print(f"[Pipeline] SegFormer import failed. Using empty condition outputs. Error: {exc}")
    get_condition_outputs = None


CONDITION_COLORS = {
    1: [255, 255, 255],  # vitiligo
    2: [255, 180, 0],    # melasma / dark spots
    3: [255, 0, 0],      # wine_stain
    4: [0, 210, 255],    # forehead wrinkle
    5: [185, 100, 255],  # crow's feet
    6: [0, 220, 120],    # nasolabial fold
}

CONDITION_LEGEND = [
    {"label": 1, "key": "vitiligo", "name": "Vitiligo", "color": "#ffffff"},
    {"label": 2, "key": "melasma", "name": "Melasma / dark spots", "color": "#ffb400"},
    {"label": 3, "key": "wine_stain", "name": "Port-wine stain", "color": "#ff0000"},
    {"label": 4, "key": "forehead_wrinkle", "name": "Forehead wrinkle", "color": "#00d2ff"},
    {"label": 5, "key": "crow_s_feet", "name": "Crow's feet", "color": "#b964ff"},
    {"label": 6, "key": "nasolabial_fold", "name": "Nasolabial fold", "color": "#00dc78"},
]


def _empty_condition_outputs(img_array: np.ndarray) -> dict:
    h, w = img_array.shape[:2]

    return {
        "condition_mask": np.zeros((h, w), dtype=np.uint8),
        "condition_map": {
            "vitiligo": {"detected": False, "area_percent": 0, "zones": []},
            "melasma": {"detected": False, "area_percent": 0, "zones": []},
            "wine_stain": {"detected": False, "area_percent": 0, "zones": []},
            "forehead_wrinkle": {"detected": False, "area_percent": 0, "zones": []},
            "crow_s_feet": {"detected": False, "area_percent": 0, "zones": []},
            "nasolabial_fold": {"detected": False, "area_percent": 0, "zones": []},
        },
    }


def _build_condition_overlay(img_array: np.ndarray, condition_mask: np.ndarray) -> dict:
    color_mask = np.zeros_like(img_array, dtype=np.uint8)

    for label_value, color in CONDITION_COLORS.items():
        color_mask[condition_mask == label_value] = color

    highlighted = condition_mask > 0
    overlay = img_array.astype(np.uint8).copy()

    if highlighted.any():
        blended = cv2.addWeighted(
            img_array.astype(np.uint8),
            0.68,
            color_mask,
            0.32,
            0,
        )
        overlay[highlighted] = blended[highlighted]

        contours_mask = highlighted.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            contours_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    success, encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    if not success:
        return {"image": None, "has_detections": bool(highlighted.any())}

    image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")

    return {
        "image": f"data:image/png;base64,{image_base64}",
        "has_detections": bool(highlighted.any()),
        "legend": [
            item
            for item in CONDITION_LEGEND
            if np.any(condition_mask == item["label"])
        ],
    }

def _debug_save_segformer_outputs(img_array, condition_mask, condition_map):
    """
    Save SegFormer runtime outputs for debugging.
    """

    debug_dir = Path(__file__).resolve().parents[1] / "debug_outputs" / "segformer"
    debug_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    mask_path = debug_dir / f"{timestamp}_condition_mask.png"
    json_path = debug_dir / f"{timestamp}_condition_map.json"
    overlay_path = debug_dir / f"{timestamp}_overlay.png"

    # Save mask as visible grayscale image
    # 0 background, 1 vitiligo, 2 melasma, 3 wine_stain
    visible_mask = (condition_mask.astype("uint8") * 80)
    cv2.imwrite(str(mask_path), visible_mask)

    # Save condition_map JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(condition_map, f, indent=2, ensure_ascii=False)

    # Create simple overlay
    overlay = img_array.copy()

    color_mask = np.zeros_like(img_array, dtype=np.uint8)

    for label_value, color in CONDITION_COLORS.items():
        color_mask[condition_mask == label_value] = color

    overlay = cv2.addWeighted(img_array.astype(np.uint8), 0.7, color_mask, 0.3, 0)

    # cv2 saves BGR, img_array is RGB, so convert
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(overlay_path), overlay_bgr)

    print("\n========== SegFormer Runtime Output ==========")
    print(json.dumps(condition_map, indent=2, ensure_ascii=False))
    print(f"condition_mask saved to: {mask_path}")
    print(f"overlay saved to:        {overlay_path}")
    print(f"condition_map saved to:  {json_path}")
    print("=============================================\n")

    return {
        "condition_mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "condition_map_path": str(json_path),
    }

async def _run_segformer_first(loop, img_array: np.ndarray) -> dict:
    if get_condition_outputs is None:
        return _empty_condition_outputs(img_array)

    try:
        return await loop.run_in_executor(None, get_condition_outputs, img_array)
    except Exception as exc:
        print(f"[Pipeline] SegFormer failed. Using empty condition outputs. Error: {exc}")
        return _empty_condition_outputs(img_array)


def _run_bisenet_and_skin_tone(img_array: np.ndarray, condition_mask: np.ndarray) -> dict:
    import torch
    from app import skin_tone_analyzer as st

    net = st._get_model()

    pil = Image.fromarray(img_array.astype(np.uint8)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    tensor = transform(pil).unsqueeze(0).to(st.DEVICE)

    with torch.no_grad():
        output = net(tensor)[0]
        parsing = output.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)

    parsing_resized = cv2.resize(
        parsing,
        (img_array.shape[1], img_array.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    return analyze_skin_tone(
        img_array,
        parsing_resized,
        condition_mask,
    )


async def run_pipeline(img_rgb, lang: str = "en") -> dict:
    img_data = preprocess(img_rgb)
    img_array = img_data["array"]

    loop = asyncio.get_running_loop()

    # 1. SegFormer runs first.
    segformer_outputs = await _run_segformer_first(loop, img_array)
    condition_mask = segformer_outputs["condition_mask"]
    condition_map = segformer_outputs["condition_map"]
    segformer_debug = _debug_save_segformer_outputs(
    img_array,
    condition_mask,
    condition_map,
)

    # 2. MediaPipe landmarks and crops.
    coords = get_landmarks(img_array)
    crops = extract_region_crops(img_array, coords)

    # 3. Claude receives condition_map as context.
    region_tasks = [
        analyze_region(
            region,
            data["base64"],
            condition_map=condition_map,
            lang=lang,
        )
        for region, data in crops.items()
    ]

    # 4. BiSeNet / skin tone receives condition_mask.
    skin_tone_task = loop.run_in_executor(
        None,
        _run_bisenet_and_skin_tone,
        img_array,
        condition_mask,
    )

    results_list, skin_tone = await asyncio.gather(
        asyncio.gather(*region_tasks),
        skin_tone_task,
    )

    region_results = {
        region: result
        for (region, _), result in zip(crops.items(), results_list)
    }

    report = build_final_report(region_results, skin_tone)

    # Keep structured SegFormer output in final JSON.
    # Do not add condition_mask because NumPy arrays are not JSON serializable.
    report["segformer_condition_map"] = condition_map
    report["condition_overlay"] = _build_condition_overlay(img_array, condition_mask)
    report["segformer_debug"] = segformer_debug

    return report
