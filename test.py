import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD IMAGE
# =========================
img = cv2.imread("/home/phatnguyen/Documents/data/glass/24_ad/test/trayng/20260407T110802.513.png")   # ảnh của bạn
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# =========================
# METHOD 1 - CLAHE
# =========================
lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(
    clipLimit=3.0,
    tileGridSize=(8,8)
)

cl = clahe.apply(l)

clahe_lab = cv2.merge((cl,a,b))
clahe_img = cv2.cvtColor(clahe_lab, cv2.COLOR_LAB2RGB)

# =========================
# METHOD 2 - HIGH PASS
# =========================
blur = cv2.GaussianBlur(img, (0,0), 15)

highpass = cv2.addWeighted(
    img,
    1.5,
    blur,
    -0.5,
    0
)

# =========================
# METHOD 3 - ORIGINAL - BLUR
# =========================
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

blur2 = cv2.GaussianBlur(gray, (0,0), 25)

sub = cv2.subtract(gray, blur2)

sub = cv2.normalize(
    sub,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

# =========================
# METHOD 4 - EDGE DETECTION
# =========================
edges = cv2.Canny(gray, 50, 150)

edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

# =========================
# METHOD 5 - FREQUENCY SEPARATION
# =========================
low = cv2.GaussianBlur(img, (0,0), 30)

high = cv2.subtract(img, low)

freq = cv2.normalize(
    high,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

# =========================
# METHOD 6 - FALSE COLOR
# =========================
false_color = cv2.applyColorMap(
    gray,
    cv2.COLORMAP_TURBO
)

false_color = cv2.cvtColor(
    false_color,
    cv2.COLOR_BGR2RGB
)

# =========================
# SHOW RESULTS
# =========================
titles = [
    "Original",
    "CLAHE",
    "High Pass",
    "Original - Blur",
    "Edge Detection",
    "Frequency Separation",
    "False Color"
]

images = [
    img,
    clahe_img,
    highpass,
    sub,
    edges_color,
    freq,
    false_color
]

plt.figure(figsize=(18,10))

for i in range(len(images)):
    plt.subplot(2,4,i+1)

    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(images[i])

    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()