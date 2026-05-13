import cv2
import numpy as np
import builtins
import PIL
from torchvision import transforms

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


def _to_2d_binary(mask):
    """Convert input mask to (H, W) binary uint8 {0,1} format."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        # If shape is (C, H, W), take the first channel
        if mask.shape[0] in (1, 3):
            mask = mask[0]
        else:
            # If shape is (H, W, C), take the first channel
            mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def _estimate_center_from_mask(mask_fg):
    """
    Estimate the center of the ring from the binary mask.
    Prefer centroid; fallback to minEnclosingCircle.
    """
    h, w = mask_fg.shape[:2]
    ys, xs = np.where(mask_fg > 0)

    if len(xs) == 0:
        return (w // 2, h // 2)

    # Centroid using moments
    m = cv2.moments(mask_fg.astype(np.uint8), binaryImage=True)
    if m["m00"] > 0:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        if np.isfinite(cx) and np.isfinite(cy):
            return (int(round(cx)), int(round(cy)))

    # Fallback: min enclosing circle
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cx, cy), _ = cv2.minEnclosingCircle(pts)
    if np.isfinite(cx) and np.isfinite(cy):
        return (int(round(cx)), int(round(cy)))

    return (w // 2, h // 2)


def _estimate_radius_band(mask_fg, center):
    """
    Estimate valid radius range of the donut region based on distance from center.
    Returns (inner_r, outer_r).
    """
    h, w = mask_fg.shape[:2]
    cx, cy = center

    ys, xs = np.where(mask_fg > 0)
    if len(xs) == 0:
        outer = 0.45 * builtins.min(h, w)
        inner = 0.30 * outer
        return float(inner), float(outer)

    dx = xs.astype(np.float32) - float(cx)
    dy = ys.astype(np.float32) - float(cy)
    d = np.sqrt(dx * dx + dy * dy)

    # Use percentile for robustness against slightly irregular boundaries
    inner_r = float(np.percentile(d, 5))
    outer_r = float(np.percentile(d, 95))

    if not np.isfinite(inner_r) or not np.isfinite(outer_r) or outer_r <= inner_r:
        inner_r = float(np.min(d))
        outer_r = float(np.max(d))

    # If too thin, expand slightly to avoid empty radius range
    if outer_r - inner_r < 2.0:
        mid = 0.5 * (inner_r + outer_r)
        inner_r = builtins.max(0.0, mid - 1.0)
        outer_r = mid + 1.0

    return inner_r, outer_r


def _downsample_max_pool(mask_l, feat_size):
    """
    Downsample using adaptive max pooling if torch is available.
    Fallback to nearest resize otherwise.
    """
    feat_size = int(feat_size)
    if feat_size <= 0:
        raise ValueError("feat_size must be a positive integer.")

    if torch is not None and F is not None:
        x = torch.from_numpy(mask_l.astype(np.float32))[None, None, ...]  # (1,1,H,W)
        y = F.adaptive_max_pool2d(x, (feat_size, feat_size))
        y = (y > 0.5).to(torch.uint8).squeeze(0).squeeze(0).cpu().numpy()
        return y.astype(np.uint8)

    # Safe fallback
    y = cv2.resize(mask_l.astype(np.uint8), (feat_size, feat_size), interpolation=cv2.INTER_NEAREST)
    return (y > 0).astype(np.uint8)


def circle(img_shape, feat_size, min_count, max_count, mask_fg):
    """
    Generate synthetic circular scratch masks on a donut-shaped object.

    Parameters
    ----------
    img_shape : tuple (C, H, W)
    feat_size : int
        Output downsampled mask size.
    min_count : int
        Minimum number of scratch rings.
    max_count : int
        Maximum number of scratch rings.
    mask_fg : np.ndarray
        Binary mask (H, W) representing the ring object.

    Returns
    -------
    mask_s : np.ndarray
        Downsampled mask (feat_size, feat_size), binary uint8 {0,1}
    mask_l : np.ndarray
        Original resolution mask (H, W), binary uint8 {0,1}
    """
    if len(img_shape) != 3:
        raise ValueError("img_shape must be (C, H, W).")

    _, H, W = img_shape
    mask_fg = _to_2d_binary(mask_fg)

    if mask_fg.shape != (H, W):
        raise ValueError(f"mask_fg must have shape {(H, W)}, but got {mask_fg.shape}.")

    min_count = int(min_count)
    max_count = int(max_count)

    if min_count < 0 or max_count < 0:
        raise ValueError("min_count/max_count must be non-negative.")

    if min_count > max_count:
        min_count, max_count = max_count, min_count

    # Initialize scratch mask
    mask_l = np.zeros((H, W), dtype=np.float32)

    # Estimate center and valid radius band
    center = _estimate_center_from_mask(mask_fg)
    inner_r, outer_r = _estimate_radius_band(mask_fg, center)

    # Random number of scratch rings
    n_scratch = np.random.randint(min_count, max_count + 1) if max_count > min_count else min_count
    if n_scratch <= 0:
        n_scratch = 1

    # Thin scratch thickness
    ring_width = builtins.max(1.0, outer_r - inner_r)
    ratio = np.random.uniform(0.02, 0.07) 
    thickness = int(np.clip(round(builtins.max(1.0, ring_width * ratio)), 1, 10))
    pad = thickness // 2 + 1

    # Safe radius range inside ring
    r_low = int(np.ceil(inner_r + pad))
    r_high = int(np.floor(outer_r - pad))

    # Fallback if invalid radius range
    if r_low > r_high:
        r_low = int(np.floor(builtins.max(inner_r + 1.0, 1.0)))
        r_high = int(np.ceil(builtins.max(outer_r - 1.0, r_low)))

    use_fallback_single = r_low > r_high

    if use_fallback_single:
        # Draw a single ring at middle radius
        r = int(round(0.5 * (inner_r + outer_r)))
        r = builtins.max(1, r)
        cv2.circle(mask_l, center, r, color=1, thickness=thickness, lineType=cv2.LINE_8)
    else:
        # Draw multiple concentric scratch rings
        sampled_radii = []
        for _ in range(n_scratch):
            if r_low == r_high:
                r = r_low
            else:
                r = np.random.randint(r_low, r_high + 1)
            sampled_radii.append(int(r))

        for r in sampled_radii:
            cv2.circle(mask_l, center, int(r), color=1, thickness=thickness, lineType=cv2.LINE_8)

    # Keep only scratches inside foreground
    mask_l = (mask_l > 0).astype(np.float32) * mask_fg.astype(np.float32)

    # Downsample using max pooling
    mask_s = _downsample_max_pool(mask_l, feat_size).astype(np.float32)

    return mask_s, mask_l


if __name__ == "__main__":
    print("Cicle script")
    
    H, W = 576, 576
    transform_mask = [
        transforms.Resize(576),
        transforms.CenterCrop(576),
        transforms.ToTensor(),
    ]
    transform_mask = transforms.Compose(transform_mask)    

    img_shape = (3, H, W)
    feat_size = 16

    mask_fg = np.zeros((H, W), dtype=np.uint8)
    
    mask_fg = PIL.Image.open("20260228T045202.836.png").convert("RGB")
    
    mask_fg = torch.ceil(transform_mask(mask_fg)[0])
    
    mask_s, mask_l = circle(img_shape, feat_size, min_count=2, max_count=4 , mask_fg=mask_fg)

    # Save masks (convert to uint8 0-255)
    cv2.imwrite(
        "mask_fg.png",
        (mask_fg.detach().cpu().numpy() * 255).astype(np.uint8)
    )
    cv2.imwrite("mask_l.png", (mask_l * 255).astype(np.uint8))
    cv2.imwrite("mask_s.png", (mask_s * 255).astype(np.uint8))

    print("Saved: mask_l.png, mask_s.png")
    print("mask_l shape:", mask_l.shape, "mask_s shape:", mask_s.shape)