import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
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
# Feature Hooks
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_padim_model(product):

    model_path = os.path.join(MODEL_DIR, f"{product}_padim_model.pth")

    checkpoint = torch.load(model_path, map_location=device)

    mean = checkpoint["mean"].cpu()
    inv_cov = checkpoint["inv_cov"].cpu()
    selected_indices = checkpoint["selected_indices"]

    return mean, inv_cov, selected_indices

# ------------------------
# Detect Anomaly
# ------------------------

def detect_anomaly(image_path, product):

    mean, inv_cov, selected_indices = load_padim_model(product)

    img = Image.open(image_path).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)

    # Feature extraction
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

    # Fast Mahalanobis
    feat = feat.squeeze(0)
    feat = feat.permute(1, 2, 0)

    diff = feat - mean
    diff = diff.unsqueeze(-2)

    dist = torch.matmul(diff, inv_cov)
    dist = torch.matmul(dist, diff.transpose(-1, -2))

    anomaly_map = torch.sqrt(dist.squeeze())

    # Upsample
    anomaly_map = F.interpolate(
        anomaly_map.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).squeeze()

    # Smooth
    smoothed_map = F.avg_pool2d(
        anomaly_map.unsqueeze(0).unsqueeze(0),
        kernel_size=11,
        stride=1,
        padding=5
    ).squeeze()

    heatmap = smoothed_map.cpu().numpy()

    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Image score
    score = np.percentile(heatmap, 99)

    print(product, "model loaded")
    print("Product:", product)
    print("Anomaly Score:", round(score, 3))

    return {
        "score": float(score),
        "heatmap": heatmap.tolist()
    }
