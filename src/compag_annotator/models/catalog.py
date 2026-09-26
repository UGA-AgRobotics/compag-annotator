"""Audited upstream model identities. Importing this module never imports ML code."""
from copy import deepcopy
from ..providers.automatic import automatic_capability

SAM2_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM3_REVISION = "2345a4ad109ac29c569da749c91d84f10dc08c40"
YOLO_REVISION = "91392240568a261d2cd6e37b1684cbfedf3cd1e5"
YOLO_VERSION = "8.4.162"

PROVIDERS = {
    "sam2": {"source_revision": SAM2_REVISION, "distribution": "SAM-2", "version": "1.0",
             "license": "Apache-2.0", "license_url": f"https://github.com/facebookresearch/sam2/blob/{SAM2_REVISION}/LICENSE",
             "source_url": f"https://github.com/facebookresearch/sam2/archive/{SAM2_REVISION}.zip",
             "torch": "2.7.1", "torchvision": "0.22.1", "capabilities": ["points", "negative_points", "box", "alternatives", "generate_proposals"]},
    "sam3": {"source_revision": SAM3_REVISION, "distribution": "sam3", "version": "0.1.0",
             "license": "SAM License", "license_url": f"https://github.com/facebookresearch/sam3/blob/{SAM3_REVISION}/LICENSE",
             "source_url": f"https://github.com/facebookresearch/sam3/archive/{SAM3_REVISION}.zip",
             "torch": "2.10.0", "torchvision": "0.25.0", "capabilities": ["points", "negative_points", "box", "alternatives"]},
    "yolo": {"source_revision": YOLO_REVISION, "distribution": "ultralytics", "version": YOLO_VERSION,
             "license": "AGPL-3.0 (or separate Enterprise terms)", "license_url": "https://www.ultralytics.com/license",
             "source_url": f"https://github.com/ultralytics/ultralytics/tree/{YOLO_REVISION}",
             "torch": "2.7.1", "torchvision": "0.22.1", "capabilities": ["infer", "train", "resume", "tiles", "classes"]},
}
for _provider, _spec in PROVIDERS.items():
    _spec["capability_details"] = {"generate_proposals": automatic_capability(_provider)}

_CATALOG = []
for size, config in [("tiny", "t"), ("small", "s"), ("base_plus", "b+"), ("large", "l")]:
    _CATALOG.append({"provider": "sam2", "architecture": f"sam2.1_hiera_{size}",
                     "name": f"SAM2.1 {size.replace('_', ' ').title()}", "recommended": size == "small",
                     "config": f"configs/sam2.1/sam2.1_hiera_{config}.yaml",
                     "filename": f"sam2.1_hiera_{size}.pt",
                     "url": f"https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_{size}.pt",
                     "sha256": None, "size_bytes": None, "digest_authority": "measured_after_download; upstream publishes no digest",
                     "checkpoint_download_allowed": True, "validation_status": "awaiting_real_checkpoint_test"})
for name, size, digest in [
    ("yolo26n-seg", 6719965, "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f"),
    ("yolo26s-seg", 23467933, "3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4"),
    ("yolo11n-seg", 6182636, "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152"),
    ("yolo11s-seg", 20669228, "1caa81c0195412efa411b632bcfb8c184939dddb6ae41f6a80c41b211ff257c3"),
]:
    _CATALOG.append({"provider": "yolo", "architecture": name, "name": name,
                     "recommended": name == "yolo26s-seg", "compatibility_alternative": name.startswith("yolo11"),
                     "filename": name + ".pt", "url": f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}.pt",
                     "sha256": digest, "size_bytes": size, "digest_authority": "GitHub official release asset digest",
                     "checkpoint_download_allowed": True, "validation_status": "awaiting_real_checkpoint_test"})
_CATALOG.append({"provider": "sam3", "architecture": "sam3-image", "name": "SAM3 image (local weights)",
                 "filename": "sam3.pt", "url": None, "sha256": None, "size_bytes": None,
                 "checkpoint_download_allowed": False, "local_weights_only": True,
                 "access_url": "https://huggingface.co/facebook/sam3",
                 "validation_status": "SAM3_WEIGHTS_NOT_SUPPLIED"})


def catalog():
    return [{**deepcopy(PROVIDERS[row["provider"]]), **deepcopy(row)} for row in _CATALOG]


def get_entry(provider, architecture):
    if provider not in PROVIDERS:
        raise ValueError("Unknown provider; choose sam2, sam3 or yolo")
    # Short UI-friendly aliases resolve to an explicit architecture in every receipt.
    aliases = {"tiny": "sam2.1_hiera_tiny", "small": "sam2.1_hiera_small",
               "base_plus": "sam2.1_hiera_base_plus", "large": "sam2.1_hiera_large"} if provider == "sam2" else {}
    architecture = aliases.get(architecture, architecture)
    architecture = architecture.removesuffix(".pt") if isinstance(architecture, str) else architecture
    for row in catalog():
        if row["provider"] == provider and row["architecture"] == architecture:
            return row
    if provider == "yolo" and architecture == "custom-seg":
        return {**deepcopy(PROVIDERS[provider]), "provider": provider, "architecture": architecture,
                "name": "Local YOLO segmenter", "url": None, "filename": None, "sha256": None}
    raise ValueError(f"Unsupported {provider} architecture: {architecture}")
