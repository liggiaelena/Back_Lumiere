# rgb_estimation.py

import json
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR / "outputs" / "skin_only.png"
OUTPUT_JSON = BASE_DIR / "outputs" / "skin_rgb.json"


def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])


def main():
    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get all non-black pixels (assuming black means no skin)
    pixels = image_rgb[np.any(image_rgb != [0, 0, 0], axis=2)]

    if len(pixels) == 0:
        raise ValueError("No skin pixels found.")

    mean_rgb = np.mean(pixels, axis=0).round().astype(int).tolist()
    median_rgb = np.median(pixels, axis=0).round().astype(int).tolist()

    result = {
        "num_skin_pixels": int(len(pixels)),
        "mean_rgb": mean_rgb,
        "mean_hex": rgb_to_hex(mean_rgb),
        "median_rgb": median_rgb,
        "median_hex": rgb_to_hex(median_rgb),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Saved:", OUTPUT_JSON)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()