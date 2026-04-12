# skin_region_extraction.py

import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MASK_PATH = BASE_DIR / "outputs" / "parsing_mask.png"
IMAGE_PATH = BASE_DIR / "outputs" / "face_crop.jpg"
OUTPUT_PATH = BASE_DIR / "outputs" / "skin_only.png"

SKIN_CLASS = 1


def main():
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(IMAGE_PATH))

    if mask is None or image is None:
        print("Error loading files")
        return

    skin_mask = (mask == SKIN_CLASS).astype(np.uint8) * 255

    skin_mask = cv2.resize(skin_mask, (image.shape[1], image.shape[0]))

    skin_only = cv2.bitwise_and(image, image, mask=skin_mask)

    cv2.imwrite(str(OUTPUT_PATH), skin_only)

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()