# Models

This directory holds the OpenVINO IR files used for inference. The model files
themselves are **not** committed (they are binary artefacts); download them
before running the application.

## Expected layout

```
models/
└── face-detection-0200/
    ├── face-detection-0200.xml
    └── face-detection-0200.bin
```

## How to download

From the project root:

```bash
python scripts/download_models.py            # FP16 (default, recommended)
python scripts/download_models.py --precision FP32
```

Or, if you have the OpenVINO development tools installed:

```bash
omz_downloader --name face-detection-0200 -o models/
```

## About `face-detection-0200`

A lightweight SSD-MobileNetV2 face detector from Intel's Open Model Zoo.

* **Input:** `1 x 3 x 256 x 256`, BGR, `uint8`
* **Output:** `1 x 1 x N x 7` — each row is
  `[image_id, label, confidence, x_min, y_min, x_max, y_max]`
  with box coordinates normalised to `[0, 1]`.

The application reads the input geometry directly from the model, so a
different-resolution variant of the same family will also work without code
changes.
