import asyncio
from app.image_utils import preprocess
from app.mediapipe_utils import get_landmarks, extract_region_crops
from app.vision import analyze_region
from app.color_utils import build_final_report
from app.skin_tone_analyzer import analyze_skin_tone
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import cv2

# We'll attempt to use any available SegFormer wrapper; if missing we
# fall back to an empty condition mask (no conditions).
try:
    from app.segmentation import get_condition_mask  # optional project helper
except Exception:
    get_condition_mask = None

async def run_pipeline(img_rgb) -> dict:
    img_data = preprocess(img_rgb)
    coords   = get_landmarks(img_data["array"])
    crops    = extract_region_crops(img_data["array"], coords)

    loop = asyncio.get_event_loop()

    # Run Claude region analysis in parallel. For skin tone we need to
    # obtain BiSeNet parsing map and a condition_mask (from SegFormer) and
    # then call analyze_skin_tone(img, bisenet_map, condition_mask) in an
    # executor to avoid blocking the event loop.
    region_tasks = [
        analyze_region(region, data["base64"])
        for region, data in crops.items()
    ]

    def _bisenet_and_condition_task(img_array):
        # Compute BiSeNet parsing map using the model loader inside
        # skin_tone_analyzer to keep weights/config centralized.
        try:
            import torch
            from app import skin_tone_analyzer as st

            net = st._get_model()

            pil = Image.fromarray(img_array)
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            tensor = transform(pil).unsqueeze(0).to(st.DEVICE)
            with torch.no_grad():
                output = net(tensor)[0]
                parsing = output.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)

            # Resize parsing back to original image size
            parsing_resized = cv2.resize(parsing, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)

        except Exception:
            # If BiSeNet inference fails, propagate by raising so caller can handle
            raise

        # Obtain condition mask (SegFormer). If helper isn't available,
        # use a conservative empty mask (no conditions).
        if get_condition_mask is not None:
            try:
                cond_mask = get_condition_mask(img_array)
            except Exception:
                cond_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
        else:
            cond_mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)

        # Call analyze_skin_tone with all required arguments
        return analyze_skin_tone(img_array, parsing_resized, cond_mask)

    bisenet_task = loop.run_in_executor(None, _bisenet_and_condition_task, img_data["array"])

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
