import os
import torch
import cv2
import numpy as np
import warnings
import PIL.Image
from torchvision import transforms
import backbones
import glass
import utils

warnings.filterwarnings('ignore')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225] 

def soft_residual_clahe(
    img,
    sigma=15,
    alpha=1.15,
    beta=0.35,
    clip_limit=2.5,
    tile_grid_size=(8,8)
):
    """
    Soft Residual + CLAHE enhancement
    for metal scratch anomaly detection

    Input:
        PIL.Image RGB

    Output:
        PIL.Image RGB
    """

    # =========================
    # PIL -> numpy
    # =========================
    img = np.array(img)

    # =========================
    # Gaussian blur
    # =========================
    blur = cv2.GaussianBlur(
        img,
        (0,0),
        sigma
    )

    # =========================
    # Residual
    # =========================
    residual = cv2.subtract(
        img,
        blur
    )

    # =========================
    # Soft blend
    # =========================
    enhanced = cv2.addWeighted(
        img,        # original
        alpha,
        residual,   # residual
        beta,
        0
    )

    enhanced = np.clip(
        enhanced,
        0,
        255
    ).astype(np.uint8)

    # =========================
    # RGB -> LAB
    # =========================
    lab = cv2.cvtColor(
        enhanced,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    # =========================
    # CLAHE on L channel
    # =========================
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    l = clahe.apply(l)

    # =========================
    # Merge LAB
    # =========================
    merged = cv2.merge((l,a,b))

    # =========================
    # LAB -> RGB
    # =========================
    out = cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2RGB
    )

    # =========================
    # numpy -> PIL
    # =========================
    out = PIL.Image.fromarray(out)

    return out


def rotate_image_reflect(image, angle):
    """Rotate a PIL image while reflecting edge pixels into new corners."""
    angle = float(angle)
    if np.isclose(angle % 360, 0.0):
        return image.copy()

    array = np.asarray(image)
    height, width = array.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        array,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return PIL.Image.fromarray(rotated)


class GLASSInference:
    def __init__(self, ckpt_path, device="cuda:0", threshold=0.5, tta_angles=(0,), tta_reduction="mean"):
        """
        Initialize the GLASS model.
        Parameters are fixed based on MVtec dataset configuration (run-custom.sh).
        
        Args:
            ckpt_path (str): Path to the checkpoint file (.pth)
            device (str): Device name ("cuda:0", "cpu", etc.)
            threshold (float): NG/OK classification threshold.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.resize = 576
        self.imagesize = 576
        

        self.tta_angles = tuple(float(angle) for angle in tta_angles)
        if not self.tta_angles:
            raise ValueError("tta_angles must contain at least one angle.")
        if tta_reduction not in {"mean", "median"}:
            raise ValueError("tta_reduction must be 'mean' or 'median'.")
        self.tta_reduction = tta_reduction

        self.resize_transform = transforms.Resize(self.resize)
        self.post_rotation_transform = transforms.Compose([
            transforms.CenterCrop(self.imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.transform = transforms.Compose([
            self.resize_transform,
            self.post_rotation_transform,
        ])

        self._build_model()
        self._load_weights(ckpt_path)

    def _build_model(self):
        # 1. Initialize backbone
        backbone_name = "wideresnet101"
        backbone = backbones.load(backbone_name)
        backbone.name = backbone_name
        backbone.seed = None 

        # 2. Initialize GLASS model
        self.model = glass.GLASS(self.device)
        self.model.load(
            backbone=backbone,
            layers_to_extract_from=["layer2", "layer3"],
            device=self.device,
            input_shape=(3, self.imagesize, self.imagesize),
            pretrain_embed_dimension=1536,
            target_embed_dimension=1536,
            patchsize=3,
            meta_epochs=100,
            eval_epochs=1,
            dsc_layers=2,
            dsc_hidden=1024,
            dsc_margin=0.5,
            train_backbone=False,
            pre_proj=1,
            mining=1,
            noise=0.015,
            radius=0.75,
            p=0.5,
            lr=0.0001,
            svd=0,
            step=40,
            limit=-1
        )
        self.model.to(self.device)

    def _load_weights(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
            
        state_dict = torch.load(ckpt_path, map_location=self.device)
        if 'discriminator' in state_dict:
            self.model.discriminator.load_state_dict(state_dict['discriminator'])
            if "pre_projection" in state_dict:
                self.model.pre_projection.load_state_dict(state_dict["pre_projection"])
            if "backbone" in state_dict:
                self.model.backbone.load_state_dict(state_dict["backbone"])
        else:
            self.model.load_state_dict(state_dict, strict=False)

        self.model.eval()

    def predict(self, image_input):
        """
        Predict for the input image.
        
        Args:
            image_input (str | np.ndarray): Image path (str) or numpy array from cv2 (BGR).
            
        Returns:
            dict: Contains the following information:
                - "prediction" (str): "NG" (defect) or "OK" (normal)
                - "confidence" (float): Model confidence score
                - "result_image" (np.ndarray): Result visualization image (3 parts, BGR numpy array)
                - "anomaly_map" (np.ndarray): Pure Heatmap image (1 channel, 0-1)
        """
        # 1. Input preprocessing
        if isinstance(image_input, str):
            image_input = PIL.Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            image_input = PIL.Image.fromarray(image_rgb)
        elif not isinstance(image_input, PIL.Image.Image):
            raise ValueError("image_input must be a path, BGR numpy array, or PIL image.")
        image_input = soft_residual_clahe(image_input)

        resized_image = self.resize_transform(image_input)
        tta_scores = []
        tta_masks = []

        for angle in self.tta_angles:
            rotated_image = rotate_image_reflect(resized_image, angle)
            input_tensor = self.post_rotation_transform(rotated_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                scores, masks = self.model._predict(input_tensor)
            tta_scores.append(float(scores[0]))
            restored_mask = rotate_image_reflect(
                PIL.Image.fromarray(np.asarray(masks[0], dtype=np.float32)),
                -angle,
            )
            tta_masks.append(np.asarray(restored_mask))

        score_values = np.asarray(tta_scores)
        score_val = float(np.mean(score_values) if self.tta_reduction == "mean" else np.median(score_values))
        mask_map = np.mean(np.stack(tta_masks), axis=0)
        
        # 3. Classify Results
        prediction = "NG" if score_val > self.threshold else "OK"
        confidence = score_val

        # 4. Data Visualization
        # Get original array (already resized by transform)
        original_img_np = utils.torch_format_2_numpy_img(
            np.array(self.transform(image_input).numpy().tolist())
        )

        mask_resized = cv2.resize(mask_map, (original_img_np.shape[1], original_img_np.shape[0]))
        mask_color = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        mask_color = (mask_color * 255).astype('uint8')
        mask_heatmap = cv2.applyColorMap(mask_color, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(original_img_np, 0.6, mask_heatmap, 0.4, 0)
        final_output = np.hstack([original_img_np, mask_heatmap, overlay])
        final_output = cv2.resize(final_output, (self.imagesize * 3, self.imagesize))
        
        label_text = f"Pred: {prediction} | Conf: {confidence}"
        text_color = (0, 0, 255) if prediction == "NG" else (0, 255, 0) # Red for NG, Green for OK
        

        return {
            "prediction": prediction,
            "confidence": confidence,
            "result_image": mask_heatmap,
            "anomaly_map": mask_map
        }

# ==========================================
# EXAMPLE USAGE IN A LARGE SYSTEM CONTEXT
# ==========================================
if __name__ == "__main__":
    # Step 1: Initialize model once when running Backend API / Background Script / Camera Socket
    print("[INFO] Initializing model...")
    model = GLASSInference(ckpt_path="/home/phatnguyen/Documents/repo/base-glass/results/models/backbone_0/mvtec_24_ad/ckpt_best_82.pth", threshold=0.5, device="cuda:0")
    print("[INFO] Model is ready!")
    
    # Step 2: Load image (Simulate image from Camera, Frontend, Folder, etc.)
    # Call .predict() whenever a new image is available
    image_path = "/home/phatnguyen/Documents/test/20260228T034228.601.png"
    if os.path.exists(image_path):
        result = model.predict(image_path)
        print(f"Analysis result: {result['prediction']} - Point: {result['confidence']}")
        
        # Save or respond with image...
        cv2.imwrite("output.png", result["result_image"])
        print("Saved result image to output.png")
