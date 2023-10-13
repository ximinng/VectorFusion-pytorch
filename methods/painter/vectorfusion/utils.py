# -*- coding: utf-8 -*-
# Author: ximing
# Description: utils
# Copyright (c) 2023, XiMing Xing.
# License: MIT License

import pathlib
from typing import Union, List, Text, BinaryIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision.utils import make_grid


def plt_batch(
        photos: torch.Tensor,
        sketch: torch.Tensor,
        step: int,
        prompt: str,
        save_path: str,
        name: str,
        dpi: int = 300
):
    if photos.shape != sketch.shape:
        raise ValueError("photos and sketch must have the same dimensions")

    plt.figure()
    plt.subplot(1, 2, 1)  # nrows=1, ncols=2, index=1
    grid = make_grid(photos, normalize=True, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title("Generated sample")

    plt.subplot(1, 2, 2)  # nrows=1, ncols=2, index=2
    grid = make_grid(sketch, normalize=False, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.title(f"Rendering result - {step} steps")

    plt.suptitle(insert_newline(prompt), fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{save_path}/{name}.png", dpi=dpi)
    plt.close()


def log_tensor_img(inputs, output_dir, output_prefix="input", norm=False, dpi=300):
    grid = make_grid(inputs, normalize=norm, pad_value=2)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    plt.imshow(ndarr)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{output_prefix}.png", dpi=dpi)
    plt.close()


def view_images(images: Union[np.ndarray, List],
                num_rows: int = 1,
                offset_ratio: float = 0.02,
                save_image: bool = False,
                fp: Union[Text, pathlib.Path, BinaryIO] = None) -> np.ndarray:
    if save_image:
        assert fp is not None

    if isinstance(images, np.ndarray) and images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images] if not isinstance(images, list) else images
        num_empty = len(images) % num_rows

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    # Calculate the composite image
    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = int(np.ceil(num_items / num_rows))  # count the number of columns
    image_h = h * num_rows + offset * (num_rows - 1)
    image_w = w * num_cols + offset * (num_cols - 1)
    assert image_h > 0, "Invalid image height: {} (num_rows={}, offset_ratio={}, num_items={})".format(
        image_h, num_rows, offset_ratio, num_items)
    assert image_w > 0, "Invalid image width: {} (num_cols={}, offset_ratio={}, num_items={})".format(
        image_w, num_cols, offset_ratio, num_items)
    image_ = np.ones((image_h, image_w, 3), dtype=np.uint8) * 255

    # Ensure that the last row is filled with empty images if necessary
    if len(images) % num_cols > 0:
        empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
        num_empty = num_cols - len(images) % num_cols
        images += [empty_images] * num_empty

    for i in range(num_rows):
        for j in range(num_cols):
            k = i * num_cols + j
            if k >= num_items:
                break
            image_[i * (h + offset): i * (h + offset) + h, j * (w + offset): j * (w + offset) + w] = images[k]

    pil_img = Image.fromarray(image_)
    if save_image:
        pil_img.save(fp)
    return pil_img
