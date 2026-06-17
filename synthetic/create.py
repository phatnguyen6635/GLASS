from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np


def crop_and_resize(image_path):
    # 1. Đọc ảnh bằng PIL (RGB = 3 kênh)
    img = Image.open(image_path).convert("RGB")

    # 2. Convert sang numpy -> tensor float32 (C, H, W)
    img_np = np.array(img)  # (H, W, C)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()  # (C, H, W)

    # 3. Crop vùng cần lấy
    x, y, w, h = 505, 847, 5, 5
    crop = img_tensor[:, y:y+h, x:x+w]  # (C, 20, 20)

    # 4. Resize lên 576x576
    crop = crop.unsqueeze(0)  # thêm batch dimension (1, C, H, W)
    resized = F.interpolate(crop, size=(576, 576), mode='bilinear', align_corners=False)
    resized = resized.squeeze(0)  # (C, 576, 576)

    # 5. Convert về PIL (nếu cần)
    resized_np = resized.permute(1, 2, 0).clamp(0, 255).byte().numpy()
    resized_img = Image.fromarray(resized_np)

    return resized_img


# Test
if __name__ == "__main__":
    out_img = crop_and_resize("/home/phatnguyen/Documents/repo/base-glass/synthetic/hard/196_20260521T040416.040.png")
    out_img.save("output.png")
