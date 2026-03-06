import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# EXACT SAME MODEL LOADING AS COLAB
resnet = models.resnet18(pretrained=True)
resnet = resnet.to(device)
resnet.eval()

features = {}

def hook_layer2(module, input, output):
    features["layer2"] = output

def hook_layer3(module, input, output):
    features["layer3"] = output

resnet.layer2.register_forward_hook(hook_layer2)
resnet.layer3.register_forward_hook(hook_layer3)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def load_padim_model(product):

    model_path = os.path.join(MODEL_DIR, f"{product}_padim_model.pth")

    checkpoint = torch.load(model_path, map_location=device)

    mean = checkpoint["mean"].cpu()
    inv_cov = checkpoint["inv_cov"].cpu()
    selected_indices = checkpoint["selected_indices"]

    return mean, inv_cov, selected_indices


def detect_anomaly(image_path, product):

    mean, inv_cov, selected_indices = load_padim_model(product)

    img = Image.open(image_path).convert("RGB")

    x = transform(img).unsqueeze(0).to(device)

    # IMPORTANT
    features.clear()

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

    feat = feat.squeeze(0)
    feat = feat.permute(1, 2, 0)

    diff = feat - mean
    diff = diff.unsqueeze(-2)

    dist = torch.matmul(diff, inv_cov)
    dist = torch.matmul(dist, diff.transpose(-1, -2))

    anomaly_map = torch.sqrt(dist.squeeze())

    anomaly_map = F.interpolate(
        anomaly_map.unsqueeze(0).unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).squeeze()

    smoothed_map = F.avg_pool2d(
        anomaly_map.unsqueeze(0).unsqueeze(0),
        kernel_size=11,
        stride=1,
        padding=5
    ).squeeze()

    heatmap = smoothed_map.cpu().numpy()

    # score from RAW map (same as colab)
    score = float(np.percentile(heatmap, 99))

    # normalize ONLY for visualization
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return {
        "score": score,
        "heatmap": heatmap.tolist()
    }
