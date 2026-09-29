from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


WASTE_RELEVANT_CLASSES = {
    "person",
    "backpack",
    "handbag",
    "suitcase",
    "bottle",
    "cup",
    "bowl",
    "dog",
    "cat",
    "chair",
    "bench",
    "potted plant",
}


@dataclass(slots=True)
class Detection:
    class_name: str
    confidence: float
    bbox: list[int]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["class"] = payload.pop("class_name")
        return payload


class YoloObjectDetector:
    """Thin wrapper around a pretrained YOLO model for frame inference."""

    def __init__(
        self,
        model_name: str | Path = "yolov8n.pt",
        device: str | None = None,
    ) -> None:
        self.model_name = str(model_name)
        self.model = YOLO(model_name)
        self.device = device
        names = self.model.names
        self.class_names = names if isinstance(names, dict) else dict(enumerate(names))

    def detect_objects(
        self,
        frame: Any,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run object detection on a single frame and return structured results."""
        results = self.model.predict(
            source=frame,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            device=self.device,
        )
        boxes = results[0].boxes
        detections: list[dict[str, Any]] = []
        if boxes is None:
            return detections

        for box in boxes:
            class_id = int(box.cls.item())
            class_name = self.class_names[class_id]
            if class_names and class_name not in class_names:
                continue

            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            confidence = float(box.conf.item())
            detection = Detection(
                class_name=class_name,
                confidence=confidence,
                bbox=[x1, y1, x2, y2],
            )
            detections.append(detection.to_dict())

        return detections

    def supported_waste_related_classes(self) -> list[str]:
        return sorted(
            class_name
            for class_name in self.class_names.values()
            if class_name in WASTE_RELEVANT_CLASSES
        )


def draw_detections(frame: Any, detections: list[dict[str, Any]]) -> Any:
    """Draw detections in-place on a frame."""
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = f'{detection["class"]} {detection["confidence"]:.2f}'
        color = (24, 190, 255) if detection["class"] == "person" else (68, 221, 119)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(
            frame,
            (x1, max(0, y1 - 24)),
            (x1 + max(110, len(label) * 8), y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    return frame


def ensure_output_path(input_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{input_path.stem}_detected.mp4"
