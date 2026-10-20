from pathlib import Path
import cv2
import numpy as np

# ====== CẤU HÌNH ======
MASK_DIR = Path("/home/avs/phatnguyen/data/fg_mask/23/")          # Mask có chấm
IMAGE_DIR = Path("/home/avs/phatnguyen/data/glass/23/test/good/")       # Ảnh thực tương ứng
OUT_MASK_DIR = Path("/home/avs/phatnguyen/data/fg_mask/out_mask/")
OUT_IMAGE_DIR = Path("/home/avs/phatnguyen/data/fg_mask/out_img/")

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def clean_mask_and_get_dots(mask):
    """Giữ vùng trắng lớn nhất; trả về mask sạch và mask chỉ chứa các chấm bị loại."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if count <= 1:
        return np.zeros_like(binary), np.zeros_like(binary)

    # Bỏ nhãn 0 (nền), giữ vùng trắng có diện tích lớn nhất
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    # Chỉ lấy các vùng trắng đã bị loại (chấm nhỏ), không lấy nền đen
    dots_mask = cv2.bitwise_and(binary, cv2.bitwise_not(clean_mask))
    return clean_mask, dots_mask


def main():
    OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    if not MASK_DIR.exists() or not IMAGE_DIR.exists():
        raise FileNotFoundError("Hãy tạo folder 'mask' và 'images' cạnh script.")

    mask_files = [p for p in MASK_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in EXTS]

    processed = 0
    for mask_path in mask_files:
        image_path = IMAGE_DIR / mask_path.name

        if not image_path.exists():
            print(f"[BỎ QUA] Không tìm thấy ảnh tương ứng: {image_path}")
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        if mask is None or image is None:
            print(f"[LỖI] Không đọc được: {mask_path.name}")
            continue

        if image.shape[:2] != mask.shape[:2]:
            print(f"[BỎ QUA] Kích thước ảnh và mask khác nhau: {mask_path.name}")
            continue

        clean_mask, dots_mask = clean_mask_and_get_dots(mask)

        # Xóa các chấm tương ứng trên ảnh thực bằng inpainting,
        # giữ nguyên toàn bộ phần ảnh còn lại.
        if np.any(dots_mask):
            if image.ndim == 3 and image.shape[2] == 4:
                # Inpaint trên RGB rồi ghép lại alpha
                rgb = image[:, :, :3]
                alpha = image[:, :, 3]
                fixed_rgb = cv2.inpaint(rgb, dots_mask, 3, cv2.INPAINT_TELEA)
                result = cv2.merge((*cv2.split(fixed_rgb), alpha))
            else:
                result = cv2.inpaint(image, dots_mask, 3, cv2.INPAINT_TELEA)
        else:
            result = image.copy()

        cv2.imwrite(str(OUT_MASK_DIR / mask_path.name), clean_mask)
        cv2.imwrite(str(OUT_IMAGE_DIR / image_path.name), result)
        processed += 1
        print(f"[OK] {mask_path.name}")

    print(f"\nHoàn tất: xử lý {processed} cặp ảnh.")
    print(f"Mask sạch: {OUT_MASK_DIR}")
    print(f"Ảnh đã xóa chấm: {OUT_IMAGE_DIR}")


if __name__ == "__main__":
    main()