"""
cached_functions.py - Кэшированные версии функций
"""

import cv2
import numpy as np
import hashlib
import os
from auto_analysis import get_foot_mask as original_get_foot_mask

CACHE_DIR = "./cache/"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_foot_mask_cached(src_image, image_hash=None):
    """Кэшированная версия get_foot_mask"""
    if image_hash is None:
        image_hash = hashlib.md5(src_image.tobytes()).hexdigest()

    cache_file = os.path.join(CACHE_DIR, f"{image_hash}_foot_mask.npy")

    if os.path.exists(cache_file):
        return np.load(cache_file)

    result = original_get_foot_mask(src_image)
    np.save(cache_file, result)
    return result