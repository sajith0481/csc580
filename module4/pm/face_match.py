#!/usr/bin/env python3
# (full script was in previous cell; rewriting full content here)
import argparse, sys
from typing import List, Tuple
try:
    import face_recognition
except ImportError as e:
    print("ERROR: Please install 'face_recognition' (and opencv-python, pillow).")
    sys.exit(1)
import cv2, numpy as np

def load_face_encodings(img_path: str):
    image = face_recognition.load_image_file(img_path)
    boxes = face_recognition.face_locations(image, model="hog")
    encodings = face_recognition.face_encodings(image, known_face_locations=boxes)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image_bgr, encodings, boxes

def draw_boxes(img, boxes, color=(0,0,255), labels=None):
    for i, (top,right,bottom,left) in enumerate(boxes):
        cv2.rectangle(img, (left, top), (right, bottom), color, 2)
        if labels and i < len(labels) and labels[i]:
            (w,h), base = cv2.getTextSize(labels[i], cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (left, top-h-base-4), (left+w+4, top), color, cv2.FILLED)
            cv2.putText(img, labels[i], (left+2, top-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--individual", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", default="annotated_output.png")
    ap.add_argument("--tolerance", type=float, default=0.60)
    args = ap.parse_args()

    ind_img, ind_encs, _ = load_face_encodings(args.individual)
    if len(ind_encs) == 0:
        print(f"[!] No face found in individual image: {args.individual}"); sys.exit(2)
    individual_encoding = ind_encs[0]

    grp_img, grp_encs, grp_boxes = load_face_encodings(args.group)
    if len(grp_encs) == 0:
        print(f"[!] No faces found in group image: {args.group}"); sys.exit(3)

    distances = face_recognition.face_distance(grp_encs, individual_encoding)
    matches = [d <= args.tolerance for d in distances]

    labels = [f"d={d:.2f}" for d in distances]
    for i, box in enumerate(grp_boxes):
        color = (0,255,0) if matches[i] else (0,0,255)
        draw_boxes(grp_img, [box], color=color, labels=[labels[i]])

    any_match = any(matches)
    best_dist = float(min(distances))
    if any_match:
        print("✅ MATCH FOUND: The individual appears in the group photo.")
    else:
        print("❌ NO MATCH: The individual does not appear in the group photo.")
    print(f"Best distance: {best_dist:.4f} (tolerance: {args.tolerance:.2f})")
    cv2.imwrite(args.out, grp_img)
    print(f"Annotated image saved to: {args.out}")

if __name__ == "__main__":
    main()
