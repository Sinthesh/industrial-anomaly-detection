import time
import json
import os

from vision_layer.detect import detect_anomaly


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, "inspection_log.txt")

os.makedirs(LOG_DIR, exist_ok=True)


def process_inspection(image_path, product):

    start_time = time.time()

    print("Starting inspection...")
    print("Product:", product)

    score, heatmap = detect_anomaly(image_path, product)

    end_time = time.time()
    runtime = round(end_time - start_time, 3)

    result = {
        "product": product,
        "image": image_path,
        "score": float(score),
        "runtime_seconds": runtime,
        "heatmap": heatmap.tolist()
    }

    log_result(result)

    print("Inspection completed in", runtime, "seconds")

    return result


def log_result(result):

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")
