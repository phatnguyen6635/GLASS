"""
Tkinter Heatmap Drawer

Features:
- Load an image
- Draw segments (freehand lines) on the image with the mouse
- The drawing is stored in a float mask (not binary), blurred to form a smooth heatmap
- Live preview overlay of heatmap (colormap) blended with the original image
- Adjustable brush size, blur strength, and heatmap alpha
- Undo last stroke, Clear all, Save overlay and raw heatmap

Dependencies:
- Python 3.8+
- Pillow (PIL)
- numpy
- opencv-python

Run:
    python tk_heatmap_app.py

"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os

class HeatmapDrawerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heatmap Drawer - Tkinter")

        # State
        self.orig_pil = None
        self.orig_cv = None  # BGR numpy
        self.display_imgtk = None
        self.mask = None  # single-channel float mask 0..255
        self.strokes = []  # list of strokes: each stroke = (points_list, thickness)

        # Default parameters
        self.brush_size = 20
        self.blur_strength = 25
        self.alpha = 0.6

        # UI layout
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(fill=tk.X)

        self.load_btn = tk.Button(self.top_frame, text="Load Image", command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=4, pady=4)

        self.save_btn = tk.Button(self.top_frame, text="Save Overlay", command=self.save_overlay, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=4)

        self.save_raw_btn = tk.Button(self.top_frame, text="Save Heatmap (raw)", command=self.save_raw_heatmap, state=tk.DISABLED)
        self.save_raw_btn.pack(side=tk.LEFT, padx=4)

        self.clear_btn = tk.Button(self.top_frame, text="Clear", command=self.clear_mask, state=tk.DISABLED)
        self.clear_btn.pack(side=tk.LEFT, padx=4)

        self.undo_btn = tk.Button(self.top_frame, text="Undo", command=self.undo_stroke, state=tk.DISABLED)
        self.undo_btn.pack(side=tk.LEFT, padx=4)

        # Sliders
        self.controls_frame = tk.Frame(root)
        self.controls_frame.pack(fill=tk.X, padx=6, pady=6)

        tk.Label(self.controls_frame, text="Brush:").grid(row=0, column=0, sticky='w')
        self.brush_slider = tk.Scale(self.controls_frame, from_=1, to=100, orient=tk.HORIZONTAL, command=self.on_brush_change)
        self.brush_slider.set(self.brush_size)
        self.brush_slider.grid(row=0, column=1, sticky='we')

        tk.Label(self.controls_frame, text="Blur:").grid(row=1, column=0, sticky='w')
        self.blur_slider = tk.Scale(self.controls_frame, from_=1, to=101, orient=tk.HORIZONTAL, command=self.on_blur_change)
        self.blur_slider.set(self.blur_strength)
        self.blur_slider.grid(row=1, column=1, sticky='we')

        tk.Label(self.controls_frame, text="Alpha:").grid(row=2, column=0, sticky='w')
        self.alpha_slider = tk.Scale(self.controls_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_alpha_change)
        self.alpha_slider.set(int(self.alpha*100))
        self.alpha_slider.grid(row=2, column=1, sticky='we')

        self.controls_frame.columnconfigure(1, weight=1)

        # Canvas
        self.canvas = tk.Canvas(root, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<ButtonPress-1>', self.on_button_press)
        self.canvas.bind('<B1-Motion>', self.on_move)
        self.canvas.bind('<ButtonRelease-1>', self.on_button_release)

        # Internal drawing track
        self._current_points = []
        self._current_canvas_items = []

    # ----------------- UI callbacks -----------------
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[('Images', '*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff'), ('All files','*.*')])
        if not path:
            return
        pil = Image.open(path).convert('RGB')
        self.orig_pil = pil
        self.orig_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        h, w = self.orig_cv.shape[:2]
        self.mask = np.zeros((h, w), dtype=np.float32)
        self.strokes = []

        self.canvas.config(width=w, height=h)
        self.display_preview()

        # enable controls
        self.save_btn.config(state=tk.NORMAL)
        self.save_raw_btn.config(state=tk.NORMAL)
        self.clear_btn.config(state=tk.NORMAL)
        self.undo_btn.config(state=tk.DISABLED)

    def save_overlay(self):
        if self.orig_cv is None:
            return
        overlay = self.build_overlay_image()
        path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG','*.png'),('JPEG','*.jpg;*.jpeg')])
        if not path:
            return
        cv2.imwrite(path, overlay)
        messagebox.showinfo('Saved', f'Saved overlay to:\n{path}')

    def save_raw_heatmap(self):
        if self.orig_cv is None:
            return
        heat = self.build_heatmap_colored()
        path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG','*.png')])
        if not path:
            return
        cv2.imwrite(path, heat)
        messagebox.showinfo('Saved', f'Saved heatmap to:\n{path}')

    def clear_mask(self):
        if self.orig_cv is None:
            return
        self.mask.fill(0)
        self.strokes = []
        self._clear_canvas_stroke_items()
        self.undo_btn.config(state=tk.DISABLED)
        self.display_preview()

    def undo_stroke(self):
        if not self.strokes:
            return
        self.strokes.pop()
        # rebuild mask
        self.mask.fill(0)
        for pts, thick in self.strokes:
            self._draw_points_to_mask(pts, thick)
        self._clear_canvas_stroke_items()
        # redraw strokes on canvas for feedback
        for pts, thick in self.strokes:
            if len(pts) > 1:
                flat = [coord for p in pts for coord in p]
                item = self.canvas.create_line(*flat, width=thick, fill='red', capstyle=tk.ROUND, smooth=True)
                self._current_canvas_items.append(item)
        if not self.strokes:
            self.undo_btn.config(state=tk.DISABLED)
        self.display_preview()

    def on_brush_change(self, val):
        self.brush_size = int(val)

    def on_blur_change(self, val):
        self.blur_strength = int(val)
        if self.blur_strength < 1:
            self.blur_strength = 1

    def on_alpha_change(self, val):
        self.alpha = int(val)/100.0

    # ----------------- Drawing callbacks -----------------
    def on_button_press(self, event):
        if self.orig_cv is None:
            return
        self._current_points = [(event.x, event.y)]

    def on_move(self, event):
        if self.orig_cv is None:
            return
        self._current_points.append((event.x, event.y))
        # draw temporary on canvas
        if len(self._current_points) > 1:
            flat = [coord for p in self._current_points for coord in p]
            # remove last temp line to avoid too many objects
            if self._current_canvas_items:
                self.canvas.delete(self._current_canvas_items[-1])
                self._current_canvas_items.pop()
            item = self.canvas.create_line(*flat, width=self.brush_size, fill='red', capstyle=tk.ROUND, smooth=True)
            self._current_canvas_items.append(item)

    def on_button_release(self, event):
        if self.orig_cv is None:
            return
        if len(self._current_points) < 2:
            self._current_points = []
            return
        pts = list(self._current_points)
        thick = self.brush_size
        self.strokes.append((pts, thick))
        # draw to mask
        self._draw_points_to_mask(pts, thick)
        self._current_points = []
        # keep the visible stroke, but allow undo
        self.undo_btn.config(state=tk.NORMAL)
        self.display_preview()

    # ----------------- Mask / rendering -----------------
    def _draw_points_to_mask(self, pts, thickness):
        # convert pts to int and clip
        h, w = self.mask.shape
        pts_i = [(max(0, min(w-1, int(x))), max(0, min(h-1, int(y)))) for (x,y) in pts]
        for i in range(len(pts_i)-1):
            p1 = pts_i[i]
            p2 = pts_i[i+1]
            cv2.line(self.mask, p1, p2, color=255.0, thickness=int(thickness), lineType=cv2.LINE_AA)

    def build_blurred_mask(self):
        # Use GaussianBlur to spread intensity; kernel size must be odd
        k = max(1, int(self.blur_strength))
        # turn k into a reasonable odd kernel size (>=1)
        kernel = k if k % 2 == 1 else k+1
        # convert to uint8 for blur and colormap
        m_uint8 = np.clip(self.mask, 0, 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(m_uint8, (kernel, kernel), 0)
        return blurred

    def build_heatmap_colored(self):
        blurred = self.build_blurred_mask()
        # apply colormap - need uint8
        colored = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
        return colored

    def build_overlay_image(self):
        colored = self.build_heatmap_colored()
        # original is BGR
        orig = self.orig_cv.copy()
        alpha = float(self.alpha)
        overlay = cv2.addWeighted(orig, 1.0, colored, alpha, 0)
        return overlay

    def display_preview(self):
        if self.orig_cv is None:
            return
        overlay = self.build_overlay_image()
        # convert BGR to RGB for PIL
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(overlay_rgb)
        self.display_imgtk = ImageTk.PhotoImage(pil)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.display_imgtk)
        # redraw current stroke items (if any)
        for item in self._current_canvas_items:
            # leave them (they're already in canvas) - no action needed
            pass

# ----------------- Run -----------------
if __name__ == '__main__':
    root = tk.Tk()
    app = HeatmapDrawerApp(root)
    root.mainloop()
