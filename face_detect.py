import cv2
import urllib.request
import openvino as ov
import numpy as np

# Download model
urllib.request.urlretrieve(
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/face-detection-0200/FP32/face-detection-0200.xml",
    "face-detection-0200.xml"
)
urllib.request.urlretrieve(
    "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/face-detection-0200/FP32/face-detection-0200.bin",
    "face-detection-0200.bin"
)

# Load model
core = ov.Core()
model = core.read_model("face-detection-0200.xml")
compiled = core.compile_model(model, "CPU")
print("✅ Face detection model loaded successfully!")

# Get model info
input_layer = compiled.input(0)
output_layer = compiled.output(0)
N, C, H, W = input_layer.shape
print(f"Model expects input: {N}x{C}x{H}x{W}")

# Open webcam
cap = cv2.VideoCapture(0)
print("✅ Webcam opened! Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    resized = cv2.resize(frame, (W, H))
    input_data = np.expand_dims(resized.transpose(2, 0, 1), 0)

    # Run inference
    result = compiled([input_data])[output_layer]

    # Draw detections
    for det in result[0][0]:
        conf = det[2]
        if conf > 0.5:
            h, w = frame.shape[:2]
            x1 = int(det[3] * w)
            y1 = int(det[4] * h)
            x2 = int(det[5] * w)
            y2 = int(det[6] * h)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1,y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Face Detection - OpenVINO", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()