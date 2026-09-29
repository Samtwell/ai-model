# WasteGuard AI - Stage 1

This folder contains the first validated step of the AI pipeline:

`Recorded video -> pretrained YOLO -> person and object detection -> annotated output video`

## Why this stack

- `Ultralytics YOLOv8n`: lightweight pretrained detector for fast iteration on recorded video.
- `OpenCV`: reliable video read/write and frame annotation.

This stage deliberately does **not** include tracking, pose estimation, action recognition, or live camera input yet.

## Project layout

```text
ai/
├── models/
├── output/
│   └── evidence/
├── src/
│   ├── __init__.py
│   ├── detector.py
│   └── main.py
├── videos/
├── README.md
└── requirements.txt
```

## Setup

From `d:\wasteguardAI`:

```powershell
py -3 -m venv ai\.venv
.\ai\.venv\Scripts\python -m pip install --upgrade pip
.\ai\.venv\Scripts\python -m pip install -r ai\requirements.txt
```

## Run

Place at least one `.mp4` file in `ai\videos\`.

Process all videos in `ai\videos\`:

```powershell
.\ai\.venv\Scripts\python ai\src\main.py
```

Process one specific file:

```powershell
.\ai\.venv\Scripts\python ai\src\main.py --input ai\videos\sample.mp4
```

## Outputs

For an input like `ai\videos\dumping_01.mp4`, the script writes:

- `ai\output\dumping_01_detected.mp4`
- `ai\output\dumping_01_detections.jsonl`
- `ai\output\dumping_01_summary.json`

The summary file reports the pretrained model name, per-class detection counts, and the waste-related classes that the model can natively recognize.

## Notes on pretrained coverage

The default COCO-pretrained YOLO model is useful for a baseline, but it does **not** cover every waste category.

Commonly available relevant classes include:

- `person`
- `bottle`
- `backpack`
- `handbag`
- `suitcase`
- `cup`
- `bowl`
- `dog`

Important gaps for WasteGuard AI usually include:

- trash bag as a dedicated class
- garbage pile
- litter
- dumpster / waste bin as a reliable disposal class
- carton, wrapper, can, plastic bag, food waste, and many site-specific waste types

Those gaps are where custom fine-tuning or a custom waste dataset will likely become necessary after we validate this baseline.
