#!/usr/bin/env python3

import os

import numpy as np
import torch
from PIL import Image, ImageOps

from diff_utils.helpers import sample_pc


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
DEFAULT_IMAGE_SIZE = 224


class ConditioningSource:
    name = None

    def __init__(self, path):
        self.path = path

    def exists(self, record):
        return self.resolve(record) is not None

    def resolve(self, record):
        raise NotImplementedError

    def load(self, record):
        return self.load_from_path(self.resolve(record))

    def load_from_path(self, path):
        raise NotImplementedError


class PointCloudConditioning(ConditioningSource):
    name = "point_cloud"

    def __init__(self, path, pc_size=1024):
        super().__init__(path)
        self.pc_size = pc_size

    def resolve(self, record):
        path = os.path.join(
            self.path,
            record["dataset"],
            record["class_name"],
            record["instance_name"],
            "sdf_data.csv",
        )
        return path if os.path.isfile(path) else None

    def load_from_path(self, path):
        return sample_pc(path, self.pc_size)


class ImageConditioning(ConditioningSource):
    name = "image"

    def resolve(self, record):
        image_dir = os.path.join(self.path, record["instance_name"])
        if not os.path.isdir(image_dir):
            return None

        image_paths = sorted(
            os.path.join(image_dir, name)
            for name in os.listdir(image_dir)
            if name.lower().endswith(IMAGE_EXTENSIONS)
        )
        return image_paths[0] if image_paths else None

    def load_from_path(self, path):
        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = resize_and_center_crop(image, DEFAULT_IMAGE_SIZE)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def resize_and_center_crop(image, size):
    width, height = image.size
    scale = size / min(width, height)
    image = image.resize((round(width * scale), round(height * scale)), Image.BICUBIC)

    left = (image.width - size) // 2
    top = (image.height - size) // 2
    return image.crop((left, top, left + size, top + size))


def build_conditioning_sources(conditioning_specs=None):
    if conditioning_specs is None:
        return []

    if isinstance(conditioning_specs, dict):
        conditioning_specs = [conditioning_specs]

    sources = []
    for spec in conditioning_specs:
        if spec["type"] == "image":
            sources.append(ImageConditioning(spec["path"]))
        elif spec["type"] == "point_cloud":
            sources.append(PointCloudConditioning(spec["path"], spec.get("pc_size", 1024)))
        else:
            raise ValueError("Unsupported conditioning type: {}".format(spec["type"]))

    return sources
