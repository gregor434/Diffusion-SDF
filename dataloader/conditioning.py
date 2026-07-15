#!/usr/bin/env python3

import os
import re
import hashlib

import numpy as np
import torch
from PIL import Image, ImageOps

from diff_utils.helpers import sample_pc


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
DEFAULT_IMAGE_SIZE = 224
DEFAULT_CLIP_MODEL = "ViT-B/32"
DEFAULT_CLIP_FEATURE_DIM = 512


_CLIP_CACHE = {}


def resolve_clip_device(clip_device):
    if clip_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if clip_device not in ("cpu", "cuda"):
        raise ValueError("clip_device must be one of: auto, cpu, cuda")
    if clip_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("clip_device='cuda' requested, but CUDA is not available")
    return clip_device


def load_clip_model(clip_model, clip_device="auto"):
    device = resolve_clip_device(clip_device)
    cache_key = (clip_model, device)
    if cache_key not in _CLIP_CACHE:
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "Image conditioning requires OpenAI CLIP. Install it with "
                "`pip install git+https://github.com/openai/CLIP.git`."
            ) from exc

        model, preprocess = clip.load(clip_model, device=device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _CLIP_CACHE[cache_key] = (model, preprocess, device)
    return _CLIP_CACHE[cache_key]


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

    def prepare(self, records, force=False):
        return False

    def is_prepared(self, record):
        return True

    def missing_prepared_records(self, records):
        return [record for record in records if not self.is_prepared(record)]


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

    def __init__(
        self,
        path,
        cache_path=None,
        clip_model=DEFAULT_CLIP_MODEL,
        clip_device="auto",
        require_cached=False,
    ):
        super().__init__(path)
        self.cache_path = cache_path or os.path.join(path, ".clip_cache")
        self.clip_model = clip_model
        self.clip_device = clip_device
        self.require_cached = require_cached
        self.prepared_with_cuda = False

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

    def load(self, record):
        cache_path = self.resolve_cache_path(record)
        if os.path.isfile(cache_path):
            return self.load_cached(cache_path)

        if self.require_cached:
            raise FileNotFoundError(
                "Missing cached CLIP image conditioning feature: {}. "
                "Run conditioning preparation before training image-conditioned models."
                .format(cache_path)
            )

        return self.encode_and_cache(record)

    def load_cached(self, cache_path):
        return torch.load(cache_path, map_location="cpu").float()

    def encode_and_cache(self, record):
        path = self.resolve(record)
        if path is None:
            raise FileNotFoundError("no image found for conditioning record: {}".format(record))

        cache_path = self.resolve_cache_path(record)
        if os.path.isfile(cache_path):
            return self.load_cached(cache_path)

        feature = self.load_from_path(path)
        self.save_cached(feature, cache_path)
        return feature

    def prepare(self, records, force=False):
        records_to_prepare = list(records) if force else self.missing_prepared_records(records)
        if not records_to_prepare:
            self.prepared_with_cuda = False
            return False

        model, preprocess, device = load_clip_model(self.clip_model, self.clip_device)
        self.prepared_with_cuda = device == "cuda"
        for record in records_to_prepare:
            path = self.resolve(record)
            if path is None:
                continue
            cache_path = self.resolve_cache_path(record)
            if os.path.isfile(cache_path) and not force:
                continue
            image = Image.open(path)
            image = ImageOps.exif_transpose(image).convert("RGB")
            feature = self.encode_image(model, preprocess, device, image)
            self.save_cached(feature, cache_path)

        return self.prepared_with_cuda

    def is_prepared(self, record):
        return os.path.isfile(self.resolve_cache_path(record))

    def resolve_cache_path(self, record):
        model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.clip_model).strip("_")
        identifier = "{}/{}/{}".format(
            record["dataset"],
            record["class_name"],
            record["instance_name"],
        )
        digest = hashlib.sha1("{}::{}".format(self.clip_model, identifier).encode("utf8")).hexdigest()[:12]
        filename = "{}-{}.pt".format(model_name, digest)
        return os.path.join(
            self.cache_path,
            record["dataset"],
            record["class_name"],
            record["instance_name"],
            filename,
        )

    def load_from_path(self, path):
        image = Image.open(path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        model, preprocess, device = load_clip_model(self.clip_model, self.clip_device)
        return self.encode_image(model, preprocess, device, image)

    def encode_image(self, model, preprocess, device, image):
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model.encode_image(image_tensor)
        return feature.cpu().float().view(1, -1).contiguous()

    def save_cached(self, feature, cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = "{}.tmp.{}".format(cache_path, os.getpid())
        torch.save(feature.cpu().float(), tmp_path)
        os.replace(tmp_path, cache_path)


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
            sources.append(ImageConditioning(
                spec["path"],
                cache_path=spec.get("cache_path"),
                clip_model=spec.get("clip_model", DEFAULT_CLIP_MODEL),
                clip_device=spec.get("clip_device", "auto"),
                require_cached=spec.get("require_cached", False),
            ))
        elif spec["type"] == "point_cloud":
            sources.append(PointCloudConditioning(spec["path"], spec.get("pc_size", 1024)))
        else:
            raise ValueError("Unsupported conditioning type: {}".format(spec["type"]))

    return sources
