import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import os


# ------------------------
# Device
# ------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------
# Image Transform
# ------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ------------------------
# Load ResNet18
# ------------------------

resnet = models.resnet18(pretrained=True)
resnet = resnet.to(device)
resnet.eval()


# ------------------------
# Feature Hook Storage
# ------------------------

features = {}


def hook_layer2(module, input, output):
    features["layer2"] = output


def hook_layer3(module, input, output):
    features["layer3"] = output


resnet.layer2.register_forward_hook(hook_layer2)
resnet.layer3.register_forward_hook(hook_layer3)


# ------------------------
# Load PaDiM Model
# ------------------------

def load_padim_model(product):

    model_path = f"/content/drive/MyDrive/industrial_defect_detection/models/{product}_padim_model.pth"

    checkpoint = torch.load(model_path)

    mean = checkpoint["mean"]
    inv_cov = checkpoint["inv_cov"]
    selected_indices = checkpoint["selected_indices"]

    return mean, inv_cov, selected_indices


# ------------------------
# Detect Anomaly
# ------------------------

def detect_anomaly(image_path, product):

    mean, inv_cov, selected_indices = load_padim_model(product)

    img = Image.open(image_path).convert("RGB")
    original_img = img.copy()

    x = transform(img).unsqueeze(0).to(device)

    # ------------------------
    # Feature Extraction
    # ------------------------

    with torch.no_grad():

        _ = resnet(x)

        layer2_feat = features["layer2"]
        layer3_feat = features["layer3"]

        layer3_feat = F.interpolate(
            layer3_feat,
            size=layer2_feat.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        feat = torch.cat([layer2_feat, layer3_feat], dim=1)

    feat = feat[:, selected_indices, :, :].cpu()

    # ------------------------
    # Fast Mahalanobis Distance
    # ------------------------

    feat = feat.squeeze(0)         # C,H,W
    feat = feat.permute(1, 2, 0)   # H,W,C

    diff = feat - mean             # H,W,C

    diff = diff.unsqueeze(-2)      # H,W,1,C

    dist = torch.matmul(diff, inv_cov)
    dist = torch.matmul(dist, diff.transpose(-1, -2))

    anomaly_map = torch.sqrt(dist.squeeze())

    # ------------------------
    # Upsample anomaly map
    # ------------------------

    anomaly_map = F.interpolate(
        anomaly_map.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).squeeze()

    # ------------------------
    # Smoothing
    # ------------------------

    smoothed_map = F.avg_pool2d(
        anomaly_map.unsqueeze(0).unsqueeze(0),
        kernel_size=11,
        stride=1,
        padding=5
    ).squeeze()

    heatmap = smoothed_map.numpy()

    # ------------------------
    # Normalize Heatmap
    # ------------------------

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # ------------------------
    # Image-level Score
    # ------------------------

    score = np.percentile(heatmap, 99)

    print(product, "model loaded")
    print("Product:", product)
    print("Anomaly Score:", round(score, 3))

    # ------------------------
    # Visualization
    # ------------------------

    plt.figure(figsize=(12,4))

    # Original Image
    plt.subplot(1,3,1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")

    # Heatmap
    plt.subplot(1,3,2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Anomaly Heatmap")
    plt.axis("off")

    # Overlay
    plt.subplot(1,3,3)
    plt.imshow(original_img.resize((224,224)))
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.title("Overlay Detection")
    plt.axis("off")

    plt.show()

    return score, heatmap