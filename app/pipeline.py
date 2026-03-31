import asyncio
from app.image_utils import preprocess
from app.mediapipe_utils import get_landmarks, extract_region_crops
from app.vision import analyze_region
from app.color_utils import build_final_report

async def run_pipeline(img_rgb) -> dict:
    img_data = preprocess(img_rgb)
    coords = get_landmarks(img_data["array"])
    crops = extract_region_crops(img_data["array"], coords)

    tasks = [
        analyze_region(region, data["base64"])
        for region, data in crops.items()
    ]
    results_list = await asyncio.gather(*tasks)

    region_results = {
        region: result
        for (region, _), result in zip(crops.items(), results_list)
    }

    return build_final_report(region_results)
