from ultralytics.data.annotator import auto_annotate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="/Users/sontung/Desktop/3.Project/SIT723/afl_frames")
parser.add_argument("--det_model", type=str, default="yolo11x.pt")
parser.add_argument("--sam_model", type=str, default="sam2_b.pt")
parser.add_argument("--output_dir", type=str, default="/Users/sontung/Desktop/3.Project/SIT723/afl_frames_auto_annotate_labels")

args = parser.parse_args()

auto_annotate(data=args.data, det_model=args.det_model, sam_model=args.sam_model, output_dir=args.output_dir)

# How to run this code
# python yolo_annotate.py --data /Users/sontung/Desktop/3.Project/SIT723/afl_frames --det_model yolo11x.pt --sam_model sam2_b.pt --output_dir /Users/sontung/Desktop/3.Project/SIT723/afl_frames_auto_annotate_labels