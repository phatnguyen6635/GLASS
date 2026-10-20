"""Run GLASS on one image. Results are written beside this script."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import backbones
import glass


# Keep these values aligned with the training command used for the checkpoint.
IMAGE_SIZE = 576
BACKBONE = "wideresnet50"
LAYERS = ["layer2", "layer3"]
EMBED_DIMENSION = 1536
PATCH_SIZE = 3
PRE_PROJ = 1


def main():
    checkpoint_path = Path("/home/avs/phatnguyen/repo/GLASS/results/models/backbone_0/mvtec_23/ckpt_best_186.pth")
    image_path = Path("/home/avs/phatnguyen/repo/test/20261005T065449.880.png")
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Model checkpoint not found: {}".format(checkpoint_path))
    if not image_path.is_file():
        raise FileNotFoundError("Input image not found: {}".format(image_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbones.load(BACKBONE)
    backbone.name, backbone.seed = BACKBONE, None

    model = glass.GLASS(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=LAYERS,
        device=device,
        input_shape=(3, IMAGE_SIZE, IMAGE_SIZE),
        pretrain_embed_dimension=EMBED_DIMENSION,
        target_embed_dimension=EMBED_DIMENSION,
        patchsize=PATCH_SIZE,
        dsc_layers=2,
        dsc_hidden=1024,
        pre_proj=PRE_PROJ,
    )

    state = torch.load(str(checkpoint_path), map_location=device)
    if "discriminator" in state:
        model.discriminator.load_state_dict(state["discriminator"])
        if PRE_PROJ:
            model.pre_projection.load_state_dict(state["pre_projection"])
        if "backbone" in state:
            model.backbone.load_state_dict(state["backbone"])
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    model.forward_modules.eval()

    original = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    image_tensor = preprocess(original).unsqueeze(0).to(device)

    with torch.inference_mode():
        scores, masks = model._predict(image_tensor)

    confidence = float(np.asarray(scores[0]).reshape(-1)[0])
    mask = np.clip(np.asarray(masks[0]) * 255, 0, 255).astype(np.uint8)

    # Debug visualization: original image with the anomaly heatmap overlaid.
    original_bgr = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2BGR)
    mask = cv2.resize(mask, (original_bgr.shape[1], original_bgr.shape[0]))
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    debug_image = cv2.addWeighted(original_bgr, 0.65, heatmap, 0.35, 0)
    cv2.putText(debug_image, "GLASS score: {:.6f}".format(confidence), (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    output_path = Path(__file__).resolve().parent / "debug.png"
    cv2.imwrite(str(output_path), debug_image)
    print("Confidence (raw anomaly score): {:.6f}".format(confidence))
    print("Debug image: {}".format(output_path))


if __name__ == "__main__":
    main()
