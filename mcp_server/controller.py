import time
import json
import os

from vision_layer.detect import detect_anomaly


# ------------------------
# Project Paths
# ------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "inspection_log.txt")

os.makedirs(LOG_DIR, exist_ok=True)


# ------------------------
# MCP Controller Function
# ------------------------

def process_inspection(image_path, product):

    start_time = time.time()

    print("Starting inspection...")
    print("Product:", product)

    # Run Vision Model
    score, heatmap, original_img = detect_anomaly(image_path, product)

    end_time = time.time()
    runtime = round(end_time - start_time, 3)

    result = {
        "product": product,
        "image": image_path,
        "score": float(score),
        "heatmap": heatmap.tolist(),   # <-- REQUIRED FOR UI
        "runtime_seconds": runtime
    }

    log_result(result)

    print("Inspection completed in", runtime, "seconds")

    return result


# ------------------------
# Logging Function
# ------------------------

def log_result(result):

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")
