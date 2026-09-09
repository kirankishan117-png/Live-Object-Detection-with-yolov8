"""
Live Object Detection with YOLOv8
==================================

Runs real-time object detection on either a webcam feed or a video file,
using Ultralytics YOLOv8, and displays results in an OpenCV window with
bounding boxes, class labels, confidence scores, and an FPS counter.

Usage
-----
Webcam (default camera):
    python detect.py --source 0

Video file:
    python detect.py --source path/to/video.mp4

Custom model:
    python detect.py --source 0 --model yolov8s.pt

Save annotated output to a file:
    python detect.py --source 0 --save output.mp4

Controls
--------
    q  -> quit
    p  -> pause / resume
    s  -> save a snapshot of the current frame

Requirements
------------
    pip install ultralytics opencv-python
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Live Object Detection with YOLOv8")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (e.g. 0, 1) or path to a video file. Default: 0",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Path/name of the YOLOv8 model weights. Default: yolov8n.pt (pretrained, COCO 80 classes)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (0-1). Default: 0.5",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on: 'cpu', 'cuda', 'cuda:0', or 'mps'. Default: auto-detect.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Restrict detection to specific class IDs (e.g. --classes 0 2 for person + car).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save the annotated output video (e.g. output.mp4). Optional.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested capture width for webcam sources. Default: 1280",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested capture height for webcam sources. Default: 720",
    )
    return parser.parse_args()


def resolve_source(source_str: str):
    """Return an int (webcam index) if source_str is a digit, else the path string."""
    return int(source_str) if source_str.isdigit() else source_str


def main():
    args = parse_args()
    source = resolve_source(args.source)
    is_webcam = isinstance(source, int)

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Opening source: {'webcam ' + str(source) if is_webcam else source}")
    cap = cv2.VideoCapture(source)

    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, input_fps, (frame_width, frame_height))
        print(f"[INFO] Saving annotated output to: {args.save}")

    window_name = "YOLOv8 Live Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused = False
    prev_time = time.time()
    fps_smooth = 0.0
    snapshot_count = 0

    print("[INFO] Starting detection. Press 'q' to quit, 'p' to pause, 's' to snapshot.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream / cannot read frame.")
                break

            # Run YOLOv8 inference on the frame
            results = model.predict(
                source=frame,
                conf=args.conf,
                classes=args.classes,
                device=args.device,
                verbose=False,
            )

            # Draw boxes/labels onto the frame
            annotated_frame = results[0].plot()

            # FPS calculation (smoothed)
            now = time.time()
            dt = now - prev_time
            prev_time = now
            current_fps = 1.0 / dt if dt > 0 else 0.0
            fps_smooth = current_fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * current_fps)

            cv2.putText(
                annotated_frame,
                f"FPS: {fps_smooth:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            # Detection count overlay
            num_detections = len(results[0].boxes)
            cv2.putText(
                annotated_frame,
                f"Objects: {num_detections}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            if writer is not None:
                writer.write(annotated_frame)

            display_frame = annotated_frame
        else:
            cv2.putText(
                display_frame,
                "PAUSED",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quit requested.")
            break
        elif key == ord("p"):
            paused = not paused
            print("[INFO] Paused." if paused else "[INFO] Resumed.")
        elif key == ord("s"):
            snapshot_count += 1
            snap_path = f"snapshot_{snapshot_count}.png"
            cv2.imwrite(snap_path, display_frame)
            print(f"[INFO] Snapshot saved: {snap_path}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
