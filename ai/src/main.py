from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2

from detector import YoloObjectDetector, draw_detections, ensure_output_path


AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_DIR = AI_ROOT / "videos"
DEFAULT_OUTPUT_DIR = AI_ROOT / "output"
DEFAULT_MODEL_PATH = AI_ROOT / "models" / "yolov8n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pretrained YOLO object detection on recorded videos."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a single MP4 file. If omitted, all ai/videos/*.mp4 files are processed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for annotated videos and detection logs.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Ultralytics model checkpoint to load.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example cpu or 0.",
    )
    return parser.parse_args()


def find_input_videos(input_path: Path | None) -> list[Path]:
    if input_path is not None:
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        return [input_path]

    videos = sorted(DEFAULT_VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(
            f"No MP4 files found in {DEFAULT_VIDEO_DIR}. Add a sample video before running."
        )
    return videos


def process_video(
    video_path: Path,
    detector: YoloObjectDetector,
    output_dir: Path,
    conf_threshold: float,
    iou_threshold: float,
) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = ensure_output_path(video_path, output_dir)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    log_path = output_dir / f"{video_path.stem}_detections.jsonl"
    summary_path = output_dir / f"{video_path.stem}_summary.json"
    class_counter: Counter[str] = Counter()

    frame_index = 0
    with log_path.open("w", encoding="utf-8") as log_file:
        while True:
            success, frame = capture.read()
            if not success:
                break

            detections = detector.detect_objects(
                frame=frame,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
            for detection in detections:
                class_counter[detection["class"]] += 1

            annotated_frame = draw_detections(frame.copy(), detections)
            writer.write(annotated_frame)

            frame_record = {
                "frame_index": frame_index,
                "timestamp_sec": round(frame_index / fps, 3),
                "detections": detections,
            }
            log_file.write(json.dumps(frame_record) + "\n")
            frame_index += 1

    capture.release()
    writer.release()

    summary = {
        "input_video": str(video_path),
        "output_video": str(output_path),
        "log_file": str(log_path),
        "model": detector.model_name,
        "fps": fps,
        "frame_count": frame_index,
        "source_frame_count": total_frames,
        "supported_waste_related_classes": detector.supported_waste_related_classes(),
        "detections_by_class": dict(class_counter.most_common()),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    detector = YoloObjectDetector(model_name=args.model, device=args.device)
    videos = find_input_videos(args.input)

    for video_path in videos:
        summary = process_video(
            video_path=video_path,
            detector=detector,
            output_dir=args.output_dir,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
