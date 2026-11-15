import numpy as np


def detect_candidate_changes(diff_map, threshold_sigma=3.0):
    mean_val = np.mean(diff_map)
    std_val = np.std(diff_map)
    threshold = mean_val + (threshold_sigma * std_val)

    y_indices, x_indices = np.where(np.abs(diff_map) > threshold)

    candidates = []
    for y, x in zip(y_indices, x_indices):
        candidates.append({
            "x": int(x),
            "y": int(y),
            "intensity_diff": float(diff_map[y, x])
        })

    return candidates, threshold


def extract_patch_features(diff_map, window_size=5):
    """
    Unsupervised ML (Isolation Forest) için bölgesel istatistiksel özellikler çıkarır.
    """
    h, w = diff_map.shape
    features = []
    coords = []

    half = window_size // 2
    for y in range(half, h - half, window_size):
        for x in range(half, w - half, window_size):
            patch = diff_map[y - half:y + half + 1, x - half:x + half + 1]
            mean_val = np.mean(patch)
            std_val = np.std(patch)
            max_val = np.max(np.abs(patch))

            features.append([mean_val, std_val, max_val])
            coords.append((x, y))

    return np.array(features), coords