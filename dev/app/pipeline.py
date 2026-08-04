import json
import logging
from pathlib import Path
from datetime import datetime
import base64

import asyncio

import cv2
import numpy as np
from app.image_utils import preprocess
from app.face_parsing_bisenet import (
    build_face_region_masks,
    extract_region_crops,
    parse_face,
)
from app.vision import analyze_region
from app.color_utils import build_final_report
from app.color_analyzer import analyze_region_colors, analyze_skin_tone
from app.face_detection import detect_and_zoom_face


logger = logging.getLogger(__name__)

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

ZONE_TO_REGION = {
    "forehead": "testa",
    "left_cheek": "bochecha_e",
    "right_cheek": "bochecha_d",
    "center_face": "nariz",
    "chin": "queixo",
}


def _condition_imperfections(condition_map: dict) -> list:
    """Make SegFormer detections visible even if Claude is unavailable."""
    output = []
    for condition, details in condition_map.items():
        if not details.get("detected"):
            continue
        area = float(details.get("area_percent", 0))
        intensity = "intenso" if area >= 15 else "moderado" if area >= 5 else "leve"
        zones = details.get("zones") or ["center_face"]
        for zone in zones:
            output.append({
                "tipo": condition,
                "intensidade": intensity,
                "regiao": ZONE_TO_REGION.get(zone, "nariz"),
                "source": "segformer",
                "area_percent": area,
            })
    return output


def _confirm_melasma_candidate(report: dict, condition_map: dict, candidates: dict):
    """Confirm a soft melasma mask only with independent region evidence."""
    candidate = candidates.get("melasma") if isinstance(candidates, dict) else None
    if not candidate or condition_map.get("melasma", {}).get("detected"):
        return None

    spot_types = {"mancha", "spot", "mancha_solar"}
    spot_regions = {
        item.get("regiao")
        for item in report.get("imperfeicoes", [])
        if str(item.get("tipo", "")).strip().lower() in spot_types
    }
    spot_regions.discard(None)
    candidate_regions = {
        ZONE_TO_REGION.get(zone)
        for zone in candidate.get("zones", [])
    }
    candidate_regions.discard(None)
    supporting_regions = spot_regions & candidate_regions

    # Requiring two independently analysed regions protects against turning a
    # single freckle, shadow, or compression artefact into a melasma result.
    if len(supporting_regions) < 2:
        return None

    condition_map["melasma"] = {
        "detected": True,
        "suspected": True,
        "source": "segformer_spot_fusion",
        "threshold": candidate["threshold"],
        "area_percent": candidate["area_percent"],
        "zones": candidate["zones"],
        "supporting_regions": sorted(supporting_regions),
    }
    return candidate["mask"]


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
        "condition_candidates": {},
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
        raise RuntimeError("SegFormer is unavailable; disease-safe color analysis cannot continue.")

    try:
        return await loop.run_in_executor(None, get_condition_outputs, img_array)
    except Exception as exc:
        raise RuntimeError(
            "SegFormer failed; disease-safe color analysis cannot continue."
        ) from exc


async def run_pipeline(img_rgb, lang: str = "en") -> dict:
    logger.info("Pipeline started (lang=%s)", lang)
    img_data = preprocess(img_rgb)
    img_array = img_data["array"]

    loop = asyncio.get_running_loop()

    # First find the largest face and crop a wider face-focused view. All
    # subsequent models analyze this zoomed image instead of the full photo.
    img_array, face_detection = await loop.run_in_executor(
        None, detect_and_zoom_face, img_array
    )
    logger.info("Face detected and cropped (MediaPipe)")
    success, face_encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    )
    if not success:
        raise RuntimeError("Could not encode the detected face image.")
    face_image = (
        "data:image/jpeg;base64,"
        + base64.b64encode(face_encoded.tobytes()).decode("ascii")
    )

    # 2. SegFormer detects disease/condition pixels on the zoomed face.
    segformer_outputs = await _run_segformer_first(loop, img_array)
    condition_mask = segformer_outputs["condition_mask"]
    condition_map = segformer_outputs["condition_map"]
    condition_candidates = segformer_outputs.get("condition_candidates", {})
    detected_conditions = [
        name for name, details in condition_map.items() if details.get("detected")
    ]
    logger.info("SegFormer condition segmentation completed (detected=%s)", detected_conditions)
    segformer_debug = _debug_save_segformer_outputs(
    img_array,
    condition_mask,
    condition_map,
)

    # 3. BiSeNet parses the face; all five masks share SegFormer's coordinates.
    parsing_map = await loop.run_in_executor(
        None,
        parse_face,
        img_array,
    )
    logger.info("BiSeNet face parsing completed")
    region_masks = build_face_region_masks(img_array, parsing_map)
    crops = extract_region_crops(img_array, region_masks)

    # 4. Analyze only parsed skin pixels outside every SegFormer condition.
    skin_tone = analyze_skin_tone(img_array, parsing_map, condition_mask)
    region_colors = analyze_region_colors(
        img_array,
        parsing_map,
        condition_mask,
        region_masks,
    )
    logger.info("Skin tone and region color analysis completed (median_hex=%s)", skin_tone.get("median_hex"))

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

    results_list = await asyncio.gather(*region_tasks)

    region_results = {
        region: result
        for (region, _), result in zip(crops.items(), results_list)
    }
    logger.info("Claude region analysis completed (%d regions)", len(region_results))

    report = build_final_report(region_results, skin_tone)
    confirmed_melasma_mask = _confirm_melasma_candidate(
        report, condition_map, condition_candidates
    )
    if confirmed_melasma_mask is not None:
        condition_mask[confirmed_melasma_mask > 0] = 2
    segformer_imperfections = _condition_imperfections(condition_map)
    existing_imperfections = report.get("imperfeicoes", [])
    report["imperfeicoes"] = existing_imperfections + segformer_imperfections

    # Keep structured SegFormer output in final JSON.
    # Do not add condition_mask because NumPy arrays are not JSON serializable.
    report["segformer_condition_map"] = condition_map
    report["condition_overlay"] = _build_condition_overlay(img_array, condition_mask)
    report["segformer_debug"] = segformer_debug
    report["face_detection"] = face_detection
    report["face_image"] = face_image
    report["face_regions"] = {
        region: {
            "bbox": data["bbox"],
            "bbox_percent": data["bbox_percent"],
            "healthy_skin_color": region_colors[region],
        }
        for region, data in crops.items()
    }
    logger.info("Pipeline finished, report built")

    return report
