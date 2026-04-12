import asyncio
from app.image_utils import preprocess
from app.mediapipe_utils import get_landmarks, extract_region_crops
from app.vision import analyze_region
from app.color_utils import build_final_report
from app.skin_tone_analyzer import analyze_skin_tone

async def run_pipeline(img_rgb) -> dict:
    img_data = preprocess(img_rgb)
    coords   = get_landmarks(img_data["array"])
    crops    = extract_region_crops(img_data["array"], coords)

    loop = asyncio.get_event_loop()

    # Run Claude region analysis + BiSeNet skin tone in parallel
    region_tasks = [
        analyze_region(region, data["base64"])
        for region, data in crops.items()
    ]
    bisenet_task = loop.run_in_executor(None, analyze_skin_tone, img_data["array"])

    results_list, skin_tone = await asyncio.gather(
        asyncio.gather(*region_tasks),
        bisenet_task,
    )

    region_results = {
        region: result
        for (region, _), result in zip(crops.items(), results_list)
    }

    report = build_final_report(region_results, skin_tone)
    return report
