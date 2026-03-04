import time
import json
import os

from vision_layer.detect import detect_anomaly


# ------------------------
# Log File
# ------------------------

LOG_PATH = "/content/drive/MyDrive/industrial_defect_detection/logs/inspection_log.txt"


# ------------------------
# MCP Controller Function
# ------------------------

def process_inspection(image_path, product):

    start_time = time.time()

    print("Starting inspection...")
    print("Product:", product)

    # Run Vision Model
    score, heatmap = detect_anomaly(image_path, product)

    end_time = time.time()
    runtime = round(end_time - start_time, 3)

    result = {
        "product": product,
        "image": image_path,
        "score": float(score),
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