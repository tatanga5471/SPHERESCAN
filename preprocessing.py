import numpy as np


def compute_image_difference(img1, img2):
    if img1.shape != img2.shape:
        min_h = min(img1.shape[0], img2.shape[0])
        min_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:min_h, :min_w]
        img2 = img2[:min_h, :min_w]

    diff = img2.astype(float) - img1.astype(float)
    return diff