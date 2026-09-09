# Live Object Detection with YOLOv8

Real-time object detection using [Ultralytics YOLOv8](https://docs.ultralytics.com/), running on either your **webcam** or a **video file**, with results shown live in an OpenCV window (bounding boxes, labels, confidence, FPS).

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

On first run, YOLOv8 will automatically download the pretrained model weights (e.g. `yolov8n.pt`, ~6MB) from Ultralytics — no manual download needed. This requires an internet connection the first time only.

## 2. Run it

**Webcam (default camera):**
```bash
python detect.py --source 0
```

**A different camera (if you have more than one):**
```bash
python detect.py --source 1
```

**A video file:**
```bash
python detect.py --source path/to/video.mp4
```

**Save the annotated output to a video file:**
```bash
python detect.py --source 0 --save output.mp4
```

**Only detect certain classes** (COCO class IDs — e.g. 0=person, 2=car, 16=dog):
```bash
python detect.py --source 0 --classes 0 2
```

**Use a bigger/more accurate model** (slower):
```bash
python detect.py --source 0 --model yolov8s.pt
# Options, smallest to largest/most accurate:
# yolov8n.pt < yolov8s.pt < yolov8m.pt < yolov8l.pt < yolov8x.pt
```

**Use your own custom-trained model:**
```bash
python detect.py --source 0 --model path/to/your_custom_model.pt
```

**Force CPU or GPU:**
```bash
python detect.py --source 0 --device cpu
python detect.py --source 0 --device cuda:0
```

## 3. Controls (while the window is focused)

| Key | Action              |
|-----|----------------------|
| `q` | Quit                  |
| `p` | Pause / resume        |
| `s` | Save a snapshot (PNG) |

## 4. All options

```bash
python detect.py --help
```

| Flag        | Default      | Description                                              |
|-------------|--------------|-----------------------------------------------------------|
| `--source`  | `0`          | Webcam index or path to a video file                      |
| `--model`   | `yolov8n.pt` | Model weights (pretrained name or path to custom `.pt`)   |
| `--conf`    | `0.5`        | Confidence threshold (0–1)                                 |
| `--device`  | auto         | `cpu`, `cuda`, `cuda:0`, or `mps`                          |
| `--classes` | all          | List of COCO class IDs to restrict detection to            |
| `--save`    | none         | Path to save annotated output video                        |
| `--width`   | `1280`       | Requested webcam capture width                             |
| `--height`  | `720`        | Requested webcam capture height                             |

## 5. Notes

- **Pretrained model (`yolov8n.pt` etc.)** detects the 80 [COCO classes](https://docs.ultralytics.com/datasets/detect/coco/) (person, car, dog, chair, laptop, etc.) out of the box — no training needed.
- **Custom model**: if you later train YOLOv8 on your own dataset (`yolo train ...`), just point `--model` at the resulting `best.pt`.
- If your webcam doesn't open, try a different `--source` index (0, 1, 2...) or check that no other app is using the camera.
- Performance depends on your hardware — `yolov8n` on a CPU typically runs real-time on most modern laptops; larger models benefit a lot from a GPU.
