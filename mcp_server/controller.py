import time
import json
import os

from vision_layer.detect import detect_anomaly


# ------------------------
# Paths
# ------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "inspection_log.txt")

os.makedirs(LOG_DIR, exist_ok=True)


# ------------------------
# MCP Controller
# ------------------------

def process_inspection(image_path, product):

    start_time = time.time()

    print("Starting inspection...")
    print("Product:", product)

    # Run vision model
    vision_result = detect_anomaly(image_path, product)

    runtime = round(time.time() - start_time, 3)

    result = {
        "product": product,
        "image": image_path,
        "score": vision_result["score"],
        "runtime_seconds": runtime,
        "heatmap": vision_result["heatmap"]
    }

    log_result(result)

    return result


# ------------------------
# Logging
# ------------------------

def log_result(result):

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")
