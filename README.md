# MimamoriFall — Fall Detection System

A research and implementation project for **fall detection** using video/camera feeds, intended for monitoring older adults or care environments. The system combines three deep learning models in a pipeline:

```text
Camera / Video → YOLO (person detection) → ViTPose (pose embedding extraction) → LSTM (Fall / Normal classification)
```

The training data primarily comes from **HAR-UP** (Human Activity Recognition — University of Porto), collected using the scripts in the `crawler/` directory.

## Demo Video

[![MimamoriFall Demo](https://img.youtube.com/vi/OCikfFIfeos/maxresdefault.jpg)](https://www.youtube.com/watch?v=OCikfFIfeos)

Watch directly: [https://www.youtube.com/watch?v=OCikfFIfeos](https://www.youtube.com/watch?v=OCikfFIfeos)

---

## Directory Structure

```text
job/
├── DETR+ViT+LSTM/     # Main pipeline: RT-DETR + ViTPose + LSTM
├── ViT+CNN+LSTM/      # Early experimental version: ViTPose skeleton + CNN + LSTM
├── crawler/          # Collect and download the HAR-UP dataset
├── paper/            # Related research papers
├── plot.py           # Plot confusion matrices and calculate evaluation metrics
├── requirements.txt  # Shared dependencies (training, crawler)
└── note.txt          # Quick notes on the scripts
```

---

## `DETR+ViT+LSTM/` — Main Pipeline

This is the main development branch, covering data preparation through real-time application deployment.

### `Method/` — Research and Training Workflow

| Directory | Purpose |
|-----------|---------|
| `Dataset Preparation/0. Labeling Timestamps/` | Assign fall/normal labels based on timestamps in CSV files |
| `Dataset Preparation/1. Manifest Creation/` | Create manifest files mapping images ↔ labels ↔ timestamps |
| `Dataset Preparation/2. BBox Detection/` | Detect person bounding boxes using RT-DETR-X or YOLO |
| `Dataset Preparation/3. Sequences Dataset/` | Assemble frame sequences, crop person images, and augment data |
| `Dataset Preparation/4. ViTPose Embeddings/` | Extract embedding vectors from ViTPose for each frame |
| `imageonly_embedded_dataset/` | Preprocessed dataset (train / val / test) |
| `Model/` | LSTM model definitions, data loaders, and the `train.py` script |

### `runs1/` … `runs5/` — Training Results

Each directory stores the checkpoints and associated files from an experimental run:

- `best.pt` — Best checkpoint
- `last.pt` — Final checkpoint
- `scaler.npz` — Embedding normalization parameters
- `history.jsonl` — Training logs

The checkpoint currently used for inference is `runs5/best.pt`.

### `MVP/` — Demo Web Application (Batch + Live Camera)

A web application for testing the pipeline on images/videos or live camera feeds.

| Component | Description |
|-----------|-------------|
| `backend/` | Flask API: load models and process images in batches or individual live frames |
| `frontend/` | React interface (Vite): upload images, view results, and test live camera input |

Inference pipeline: RT-DETR → person cropping → ViTPose embeddings → 10-frame buffer → LSTM classification.

### `MVP2_Live/` — Real-Time Fall Alert System

An upgraded version of the MVP focused on **real-time alerts**:

- Flask backend (port 5002) + React frontend (port 5174)
- An FSM (finite state machine) manages the alert logic: detect a fall → monitor bounding box stability for 5 seconds → trigger an agent to send a notification
- For setup instructions and API details, see [`DETR+ViT+LSTM/MVP2_Live/README.md`](DETR+ViT+LSTM/MVP2_Live/README.md)

---

## `ViT+CNN+LSTM/` — Early Experimental Version

The initial approach extracts skeletons using ViTPose, renders them on a black background, and feeds them into a **CNN + LSTM** model for classification.

| File / Directory | Purpose |
|------------------|---------|
| `prepare_labels.py` | Assign labels based on timestamps in CSV files |
| `extract_vitpose_skeletons.py` | Extract skeleton images |
| `manifestcreation.ipynb` | Create a manifest mapping labels ↔ images |
| `sequence_data.py` | Load and prepare frame sequences for training |
| `model.py` | Define `SkeletonImageLSTMClassifier` (CNN + LSTM) |
| `train_vitpose_lstm.py` | Training and inference script |
| `utils.py` | Training/evaluation utility functions |
| `mvp/` | Real-time demo using OpenCV (camera → skeleton overlay → classifier) |

For instructions on running the MVP, see [`ViT+CNN+LSTM/mvp/README.md`](ViT+CNN+LSTM/mvp/README.md).

---

## `crawler/` — HAR-UP Dataset Collection

Scripts for automatically crawling and downloading data from the HAR-UP website:

| File | Purpose |
|------|---------|
| `crawl_har_up.py` | Crawl dataset links from the website |
| `crawl_csv_har_up.py` | Crawl CSV file links |
| `download_har_up_datasets.py` | Download CSV files using the list in `har_up_dataset_links.json` |
| `har_up_links.json` / `har_up_dataset_links.json` | Lists of crawled links |

---

## `paper/`

Contains related research material (`upfall.pdf`).

---

## Root Directory Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Shared Python dependencies (PyTorch, transformers, ultralytics, selenium, etc.) |
| `plot.py` | Plot a confusion matrix heatmap and calculate Accuracy / Precision / Recall / F1 for the Fall class |
| `confusion_matrix.png` | Exported evaluation results |
| `note.txt` | Quick notes on the roles of scripts in `ViT+CNN+LSTM/` |

---

## Overall Workflow

```text
1. Collect data                   crawler/
2. Label data and create manifests Method/Dataset Preparation/ (or ViT+CNN+LSTM/)
3. Detect people (bounding boxes)  RT-DETR-X
4. Extract pose embeddings        ViTPose
5. Train the LSTM                 Method/Model/train.py → runs*/
6. Deploy inference               MVP/ or MVP2_Live/
```

---

## System Requirements

- Python 3.10+
- PyTorch (CUDA / MPS / CPU)
- OpenMMLab stack for ViTPose (see `DETR+ViT+LSTM/MVP/backend/setup_env.sh`)
- Node.js 18+ (for the MVP / MVP2_Live frontend)
