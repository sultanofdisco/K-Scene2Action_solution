"""Image binding, region preservation and conservative OCR quality handling."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import math
import logging

from f1.bindings import BindingError, verify_image_bytes
from f1.schemas import ImageRef, OCRRegion, OCRResult, OCRStatus, ReasonCode

PREPROCESSING_VERSION = "exif-rgb-white-png-v1"


@dataclass(frozen=True)
class OCRPolicy:
    min_recognition_score: float = 0.8
    max_regions: int = 256
    max_characters: int = 8192
    max_pixels: int = 16_000_000
    max_file_bytes: int = 32_000_000

    def __post_init__(self):
        if not 0 <= self.min_recognition_score <= 1:
            raise ValueError("Invalid recognition quality threshold")
        for name in ("max_regions", "max_characters", "max_pixels", "max_file_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Invalid {name}")


@dataclass(frozen=True)
class PreparedImage:
    source: ImageRef
    inspected: ImageRef
    png: bytes


def prepare_image(content: bytes, expected: ImageRef | None = None,
                  policy: OCRPolicy = OCRPolicy()) -> PreparedImage:
    from PIL import Image, ImageOps, UnidentifiedImageError
    if expected is not None:
        verify_image_bytes(expected, content)
    if len(content) > policy.max_file_bytes:
        raise BindingError(ReasonCode.OCR_BUDGET_EXCEEDED, "Image file exceeds budget")
    try:
        with Image.open(BytesIO(content)) as original:
            if original.format not in {"PNG", "JPEG", "WEBP"} or getattr(original, "n_frames", 1) != 1:
                raise BindingError(ReasonCode.IMAGE_DECODE_ERROR, "Only static PNG/JPEG/WebP supported")
            width, height = original.size
            if width * height > policy.max_pixels:
                raise BindingError(ReasonCode.OCR_BUDGET_EXCEEDED, "Image pixel budget exceeded")
            if expected is not None and (expected.width, expected.height) != (width, height):
                raise BindingError(ReasonCode.IMAGE_MISMATCH, "Decoded dimensions differ")
            source = expected or ImageRef(asset_id="source-"+sha256(content).hexdigest(),
                                         sha256=sha256(content).hexdigest(), width=width, height=height)
            rgba = ImageOps.exif_transpose(original).convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
            stream = BytesIO()
            image.save(stream, format="PNG")
            png = stream.getvalue()
            inspected = ImageRef(asset_id="inspected-"+sha256(png).hexdigest(),
                                 sha256=sha256(png).hexdigest(), width=image.width, height=image.height)
            return PreparedImage(source, inspected, png)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        if isinstance(exc, BindingError):
            raise
        raise BindingError(ReasonCode.IMAGE_DECODE_ERROR, "Image cannot be decoded") from exc


def bounds(poly):
    return min(x for x, y in poly), min(y for x, y in poly), max(x for x, y in poly), max(y for x, y in poly)


def group_polygons(polygons):
    """Conservative nearby-line components, never concatenating raw text."""
    boxes = [bounds(p) for p in polygons]
    parents = list(range(len(polygons)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for i, a in enumerate(boxes):
        for j in range(i):
            b = boxes[j]
            ha, hb = a[3]-a[1], b[3]-b[1]
            overlap = min(a[2], b[2])-max(a[0], b[0])
            gap = max(a[1], b[1])-min(a[3], b[3])
            angle_a = math.atan2(polygons[i][1][1]-polygons[i][0][1], polygons[i][1][0]-polygons[i][0][0])
            angle_b = math.atan2(polygons[j][1][1]-polygons[j][0][1], polygons[j][1][0]-polygons[j][0][0])
            if (min(ha, hb) > 0 and max(ha, hb) <= 2*min(ha, hb)
                    and overlap >= .4*min(a[2]-a[0], b[2]-b[0])
                    and 0 <= gap <= 1.5*max(ha, hb)
                    and abs(angle_a-angle_b) <= math.radians(10)):
                parents[find(i)] = find(j)
    names = {}
    return [names.setdefault(find(i), f"g{len(names):04d}") for i in range(len(polygons))]


def crop_region(image, polygon):
    import cv2
    import numpy as np
    points = np.asarray(polygon, dtype=np.float32)
    if not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < 1:
        return None
    width = round(max(np.linalg.norm(points[0]-points[1]), np.linalg.norm(points[2]-points[3])))
    height = round(max(np.linalg.norm(points[0]-points[3]), np.linalg.norm(points[1]-points[2])))
    if width < 2 or height < 2:
        return None
    dest = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, dest), (width, height),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if height / width >= 1.5:
        crop = np.rot90(crop)
    return np.ascontiguousarray(crop)


def extract_regions(prepared: PreparedImage, backend, policy: OCRPolicy = OCRPolicy()) -> OCRResult:
    import numpy as np
    from PIL import Image
    ref = prepared.inspected
    def result(status, regions=(), reasons=()):
        return OCRResult(image=ref, status=status, regions=tuple(regions), reason_codes=tuple(reasons),
                         model_version=backend.version)
    with Image.open(BytesIO(prepared.png)) as image:
        bgr = np.asarray(image)[:, :, ::-1].copy()
    try:
        raw_polygons = backend.detect(bgr)
        polygons = []
        for polygon in raw_polygons:
            points = tuple(tuple(float(c) for c in point) for point in polygon)
            if len(points) != 4 or any(len(point) != 2 for point in points):
                raise ValueError("Detector must return quadrilaterals")
            if any(not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= ref.width or not 0 <= y <= ref.height for x, y in points):
                raise ValueError("Detector coordinates exceed inspected image")
            polygons.append(points)
        polygons.sort(key=lambda p: (bounds(p)[1], bounds(p)[0]))
    except Exception as exc:
        logging.getLogger(__name__).error("OCR detection failed (%s)", type(exc).__name__)
        return result(OCRStatus.ERROR, reasons=(ReasonCode.OCR_ERROR,))
    if not polygons:
        return result(OCRStatus.COMPLETE, reasons=(ReasonCode.NO_TEXT_DETECTED,))
    over_budget = len(polygons) > policy.max_regions
    groups = [f"g{i:04d}" for i in range(len(polygons))] if over_budget else group_polygons(polygons)
    regions = [OCRRegion(region_id=f"r{i:04d}", polygon=p, raw_text="", recognition_score=None,
                         group_id=groups[i], reading_order=i) for i, p in enumerate(polygons)]
    if over_budget:
        return result(OCRStatus.INCOMPLETE, regions, (ReasonCode.OCR_BUDGET_EXCEEDED,))
    try:
        crops, indices = [], []
        for i, polygon in enumerate(polygons):
            crop = crop_region(bgr, polygon)
            if crop is not None:
                indices.append(i)
                crops.append(crop)
        outputs = backend.recognize(crops) if crops else []
        if len(outputs) != len(crops):
            raise ValueError("Missing recognition results")
        for i, (text, score) in zip(indices, outputs):
            # Revalidate, do not silently coerce a malformed SDK response.
            regions[i] = OCRRegion(**{**regions[i].model_dump(), "raw_text": text, "recognition_score": score})
    except Exception as exc:
        logging.getLogger(__name__).error("OCR recognition failed (%s)", type(exc).__name__)
        return result(OCRStatus.ERROR, regions, (ReasonCode.OCR_ERROR,))
    reasons = []
    if any(not r.raw_text.strip() or r.recognition_score is None or r.recognition_score < policy.min_recognition_score for r in regions):
        reasons.append(ReasonCode.OCR_UNREADABLE)
    if sum(len(r.raw_text) for r in regions) > policy.max_characters:
        reasons.append(ReasonCode.OCR_BUDGET_EXCEEDED)
    return result(OCRStatus.INCOMPLETE if reasons else OCRStatus.COMPLETE, regions, reasons)
