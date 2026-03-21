"""
Real-time Face Detection using OpenVINO AUTO plugin
====================================================
Automatically selects the best available device (CPU, GPU, NPU)
for inference using the OpenVINO AUTO plugin.

Author: Saeed Fahim (github.com/ffaahhimm)
GSoC 2026 - OpenVINO Toolkit
"""

import cv2
import numpy as np
import openvino as ov
import time


# ── Configuration ──────────────────────────────────────────────────────────
MODEL_XML   = "face-detection-0200.xml"
DEVICE      = "AUTO"          # AUTO, CPU, GPU, NPU
THRESHOLD   = 0.5             # Confidence threshold
INPUT_SIZE  = (256, 256)      # Model input size


def preprocess(frame, input_size):
    """Resize and transpose frame for model input."""
    resized = cv2.resize(frame, input_size)
    blob = np.expand_dims(resized.transpose(2, 0, 1), axis=0)
    return blob.astype(np.float32)


def draw_detections(frame, detections, threshold):
    """Draw bounding boxes and confidence scores on frame."""
    h, w = frame.shape[:2]
    count = 0

    for detection in detections[0][0]:
        confidence = float(detection[2])
        if confidence < threshold:
            continue

        x1 = int(detection[3] * w)
        y1 = int(detection[4] * h)
        x2 = int(detection[5] * w)
        y2 = int(detection[6] * h)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw confidence score
        label = f"{confidence:.0%}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        count += 1

    return frame, count


def main():
    # ── Load model ─────────────────────────────────────────────────────────
    print(f"Loading model on device: {DEVICE}")
    core = ov.Core()

    # Print available devices
    available = core.available_devices
    print(f"Available devices: {available}")

    model = core.read_model(MODEL_XML)
    compiled_model = core.compile_model(model, DEVICE)

    # Get actual device being used
    exec_device = compiled_model.get_property("EXECUTION_DEVICES")
    print(f"Inference running on: {exec_device}")

    infer_request = compiled_model.create_infer_request()
    output_layer  = compiled_model.output(0)

    # ── Open webcam ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press Q to quit")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Inference ──────────────────────────────────────────────────────
        blob = preprocess(frame, INPUT_SIZE)
        infer_request.infer({0: blob})
        detections = infer_request.get_output_tensor(0).data

        # ── Post-process ───────────────────────────────────────────────────
        frame, face_count = draw_detections(frame, detections, THRESHOLD)

        # ── FPS counter ────────────────────────────────────────────────────
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        # ── Display info overlay ───────────────────────────────────────────
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(frame, f"Device: {exec_device}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, f"Faces: {face_count}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(frame, "Press Q to quit",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("OpenVINO Face Detection - AUTO Device", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
