# test_random_particle_mask.py

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt


# ============================================================
# RANDOM PARTICLE GENERATOR
# ============================================================

def generate_random_particles(
    mask_fg,
    num_particles=(3, 10),
    particle_radius=(2, 5),
    irregularity=(0.6, 1.4),
    elongation=(0.7, 1.5),
):
    """ 
    Random các hạt trên toàn bộ bề mặt sản phẩm.

    Mỗi hạt có:
        - vị trí random
        - kích thước random
        - hình dạng random
        - góc random

    Không dùng:
        - Perlin
        - grid
        - boundary
        - pattern
    """

    # --------------------------------------------------------
    # Tensor -> numpy
    # --------------------------------------------------------

    if isinstance(mask_fg, torch.Tensor):
        mask_fg = (
            mask_fg
            .detach()
            .cpu()
            .numpy()
        )

    mask_fg = np.asarray(mask_fg)
    mask_fg = np.squeeze(mask_fg)

    if mask_fg.ndim != 2:
        raise ValueError(
            f"mask_fg must be [H,W], got {mask_fg.shape}"
        )

    mask_fg = (
        mask_fg > 0
    ).astype(np.uint8)

    H, W = mask_fg.shape

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    particle_mask = np.zeros(
        (H, W),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Tất cả pixel trên sản phẩm
    # --------------------------------------------------------

    ys, xs = np.where(
        mask_fg > 0
    )

    if len(xs) == 0:
        raise ValueError(
            "mask_fg không có foreground!"
        )

    # --------------------------------------------------------

    # Random số lượng hạt
    # --------------------------------------------------------

    n_particles = np.random.randint(
        num_particles[0],
        num_particles[1] + 1
    )


    # ========================================================
    # CREATE PARTICLES
    # ========================================================

    for _ in range(n_particles):

        # ----------------------------------------------------
        # RANDOM POSITION
        # ----------------------------------------------------

        idx = np.random.randint(
            0,
            len(xs)
        )

        cx = int(xs[idx])
        cy = int(ys[idx])

        # ----------------------------------------------------
        # RANDOM SIZE
        # ----------------------------------------------------

        r = np.random.uniform(
            particle_radius[0],
            particle_radius[1]
        )

        # ----------------------------------------------------
        # RANDOM SHAPE
        # ----------------------------------------------------

        n_points = np.random.randint(
            5,
            12
        )

        points = []

        # Random rotation
        rotation = np.random.uniform(
            0,
            2 * np.pi
        )

        # Random elongation
        elong = np.random.uniform(
            elongation[0],
            elongation[1]
        )

        for i in range(n_points):

            # Góc cơ bản
            theta = (
                2.0
                * np.pi
                * i
                
            )

            # Random angle disturbance
            theta += np.random.uniform(
                -0.35,
                0.35
            )

            theta += rotation

            # Random radius
            rr = r * np.random.uniform(
                irregularity[0],
                irregularity[1]
            )

            # Ellipse nhẹ
            x = (
                cx
                + np.cos(theta)
                * rr
                * elong
            )

            y = (
                cy
                + np.sin(theta)
                * rr
            )

            x = int(np.round(x))
            y = int(np.round(y))

            x = np.clip(
                x,
                0,
                W - 1
            )

            y = np.clip(
                y,
                0,
                H - 1
            )

            points.append(
                [x, y]
            )

        points = np.asarray(
            points,
            dtype=np.int32
        )

        # ----------------------------------------------------
        # Fill particle
        # ----------------------------------------------------

        cv2.fillPoly(
            particle_mask,
            [points],
            1.0
        )

    # ========================================================
    # CHỈ GIỮ HẠT TRÊN SẢN PHẨM
    # ========================================================

    particle_mask *= mask_fg

    return particle_mask


# ============================================================
# BURR MASK
# ============================================================

def burr_mask(
    img_shape,
    feat_size,
    mask_fg,
    num_particles=(3, 10),
    particle_radius=(10, 20),
    irregularity=(0.6, 1.4),
    elongation=(0.7, 1.5),
):
    """
    Return:

        mask_s:
            [feat_size, feat_size]

        mask_l:
            [H, W]
    """

    C, H, W = img_shape

    # --------------------------------------------------------
    # Generate random particles
    # --------------------------------------------------------

    mask_l = generate_random_particles(
        mask_fg=mask_fg,
        num_particles=num_particles,
        particle_radius=particle_radius,
        irregularity=irregularity,
        elongation=elongation,
    )

    mask_l = torch.from_numpy(
        mask_l
    ).float()

    # --------------------------------------------------------
    # Check full resolution
    # --------------------------------------------------------

    if tuple(mask_l.shape) != (H, W):
        raise RuntimeError(
            f"mask_l sai shape: "
            f"{mask_l.shape}, "
            f"expected {(H, W)}"
        )

    # ========================================================
    # FULL RESOLUTION
    # [H,W]
    #
    # ->
    #
    # FEATURE RESOLUTION
    # [feat_size,feat_size]
    # ========================================================

    mask_s = F.adaptive_max_pool2d(
        mask_l.unsqueeze(0).unsqueeze(0),
        output_size=(
            feat_size,
            feat_size
        )
    )

    mask_s = mask_s[0, 0]

    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    if tuple(mask_s.shape) != (
        feat_size,
        feat_size
    ):
        raise RuntimeError(
            f"mask_s sai shape: "
            f"{mask_s.shape}, "
            f"expected {(feat_size, feat_size)}"
        )

    return (
        mask_s.numpy(),
        mask_l.numpy()
    )
