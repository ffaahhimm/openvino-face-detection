# OpenVINO Face Detection 🎯

Real-time face detection using Intel's OpenVINO toolkit and the `face-detection-0200` model — running on CPU with live webcam feed.

> Built as part of my GSoC 2026 preparation for the OpenVINO Toolkit organization.

---

## Demo

![Face Detection Demo](https://raw.githubusercontent.com/ffaahhimm/openvino-face-detection/main/demo.png)

*Real-time face detection with bounding boxes and confidence scores*

---

## What It Does

- 🔍 Loads Intel's `face-detection-0200` pretrained model
- 📷 Runs real-time inference on live webcam feed
- 🟩 Draws bounding boxes with confidence scores on detected faces
- ⚡ Runs efficiently on CPU using OpenVINO Runtime
- 🖥️ Works on Linux, Windows, and macOS

---

## Requirements

- Python 3.8+
- Webcam

---

## Installation

```bash
# Clone the repo
git clone https://github.com/ffaahhimm/openvino-face-detection.git
cd openvino-face-detection

# Install dependencies
pip install openvino opencv-python numpy
```

---

## Download the Model

```bash
# Install OpenVINO Model Downloader
pip install openvino-dev

# Download face-detection-0200
omz_downloader --name face-detection-0200
```

---

## Run

```bash
python3 face_detect.py
```

Press **Q** to quit.

---

## How It Works

```
Webcam Frame
     │
     ▼
OpenVINO Preprocessing
     │
     ▼
face-detection-0200 Model (CPU)
     │
     ▼
Post-processing (NMS + threshold)
     │
     ▼
Draw Bounding Boxes + FPS
     │
     ▼
Display Output
```

---

## Project Structure

```
openvino-face-detection/
├── face_detect.py       # Main detection script
├── README.md
└── .gitignore
```

---

## Related Work

This project is a foundation for my GSoC 2026 proposal:
**"Continuous Face-Detection with Automatic Device Switching on AI PCs using OpenVINO AUTO feature"**

The GSoC project extends this by:
- Adding GPU and NPU support via OpenVINO AUTO plugin
- Implementing runtime device switching without interrupting inference
- Building a GUI for manual and automatic device control

---

## Author

**Saeed Fahim**
- GitHub: [@ffaahhimm](https://github.com/ffaahhimm)
- OpenVINO Contributor: [PR #34743](https://github.com/openvinotoolkit/openvino/pull/34743)

---

## License

Apache 2.0
