"""Pinned PaddleOCR detectors and recognizers, kept separate to retain failed regions."""
from hashlib import sha256
import json
from pathlib import Path

MODELS = {"det": "PP-OCRv5_server_det", "rec": "korean_PP-OCRv5_mobile_rec"}


def file_hash(path):
    import hashlib
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_models(output):
    from huggingface_hub import HfApi, snapshot_download
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    lock = {"format_version": 1, "models": {}}
    for role, name in MODELS.items():
        repo = f"PaddlePaddle/{name}"
        revision = HfApi().model_info(repo, timeout=30).sha
        folder = output / role
        snapshot_download(repo, revision=revision, local_dir=folder,
                          allow_patterns=["inference.*", "config.json"])
        files = {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()}
        if not {"inference.json", "inference.pdiparams", "inference.yml"} <= files.keys():
            raise ValueError(f"Incomplete model: {name}")
        lock["models"][role] = {"name": name, "repo": repo, "revision": revision,
                                "directory": role, "files": files}
    (output / "model-lock.json").write_text(json.dumps(lock, indent=2)+"\n", encoding="utf-8")
    return lock


class PaddleBackend:
    def __init__(self, model_root, cpu_threads=4, det_limit=1536):
        root = Path(model_root).resolve()
        lock = json.loads((root / "model-lock.json").read_text(encoding="utf-8"))
        if lock["format_version"] != 1:
            raise ValueError("Unsupported model lock")
        paths = {}
        for role, expected in MODELS.items():
            entry = lock["models"][role]
            if entry["name"] != expected or entry["repo"] != f"PaddlePaddle/{expected}":
                raise ValueError("Unexpected OCR model")
            folder = (root / entry["directory"]).resolve()
            if not folder.is_relative_to(root):
                raise ValueError("Model path escapes model root")
            if not {"inference.json", "inference.pdiparams", "inference.yml"} <= entry["files"].keys():
                raise ValueError("Incomplete OCR model lock")
            for name, expected_hash in entry["files"].items():
                path = (folder / name).resolve()
                if not path.is_relative_to(folder) or file_hash(path) != expected_hash:
                    raise ValueError("OCR model file mismatch")
            paths[role] = str(folder)
        from paddleocr import TextDetection, TextRecognition
        from importlib.metadata import version
        self.version = "paddleocr-" + version("paddleocr") + ":" + sha256(
            json.dumps({"lock": lock, "paddle": version("paddlepaddle"), "det_limit": det_limit,
                        "cpu_threads": cpu_threads, "mkldnn": False}, sort_keys=True).encode()
        ).hexdigest()[:16]
        self.lock = lock
        common = dict(device="cpu", cpu_threads=cpu_threads, enable_mkldnn=False)
        self.detector = TextDetection(model_name=MODELS["det"], model_dir=paths["det"], **common)
        self.recognizer = TextRecognition(model_name=MODELS["rec"], model_dir=paths["rec"], **common)
        self.det_limit = det_limit

    def detect(self, image):
        result = list(self.detector.predict(image, batch_size=1, limit_side_len=self.det_limit,
                                           limit_type="max", thresh=0.3, box_thresh=0.6, unclip_ratio=1.5))
        if len(result) != 1:
            raise ValueError("Expected exactly one detector result")
        return result[0]["dt_polys"]

    def recognize(self, crops):
        results = list(self.recognizer.predict(crops, batch_size=8))
        if len(results) != len(crops):
            raise ValueError("Recognition count mismatch")
        return [(r["rec_text"], float(r["rec_score"])) for r in results]
