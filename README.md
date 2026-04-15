# VehiReID-RAG
**Vision–Language Framework for Vehicle Re-Identification**

VehiReID-RAG is a multimodal framework for vehicle re-identification built around fine-grained visual classification, reranking, and inference utilities for the CompCars dataset. The current public codebase provides dataset indexing, training, evaluation, and prediction pipelines centred on a ViT-based multi-task model.

---

## Overview

This repository includes code to:

- Build a CompCars index from the official dataset split files  
- Train a multi-task vehicle classification model  
- Evaluate a saved checkpoint  
- Run inference on a single image or a directory of images  

---

## Repository Structure
```text
VehiReID-RAG/
│── config/
│   └── default.yaml
│── scripts/
│   └── build_compcars_index.py
│── src/
│   ├── datasets/
│   │   └── compcars_dataset.py
│   ├── models/
│   │   └── vit_multitask.py
│   ├── eval.py
│   ├── losses.py
│   ├── mats.py
│   ├── metrics.py
│   ├── predict.py
│   ├── samplers.py
│   ├── train.py
│   └── utils.py
│── README.md
│── requirements.txt
```
---

## Requirements

- Python 3.10+
- PyTorch and dependencies from requirements.txt
- CompCars dataset (manual download required)
- GPU recommended (2 GPUs required for distributed training)

---

## Dataset

This project uses the CompCars dataset.

Official instructions:  
https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt

### Important Note
- The dataset is for non-commercial research only
- Do NOT upload or redistribute the dataset

### Expected Directory Structure
```text
data/
└── compcars/
    ├── image/
    ├── label/
    ├── train_test_split/
    ├── part/
    └── attribute/
```
---

## Installation

1. Clone the repository
git clone https://github.com/IsmailMehar/VehiReID-RAG.git
cd VehiReID-RAG

2. Create a virtual environment

Linux / macOS
python -m venv venv
source venv/bin/activate

Windows
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

---

## Data Preparation

Before training or evaluation, build dataset indexes:

python scripts/build_compcars_index.py

This generates:

data/compcars/indexes/index.csv  
data/compcars/indexes/vocabs.json  

---

## Training

Default training:
python src/train.py

Uses config:
./config/default.yaml

Multi-GPU training:
torchrun --nproc_per_node=2 src/train.py

Outputs:
runs/best.pt

---

## Evaluation

Standard:
python src/eval.py --ckpt ./runs/best.pt

Multi-GPU:
torchrun --nproc_per_node=2 src/eval.py --ckpt ./runs/best.pt

With reranking:
python src/eval.py --ckpt ./runs/best.pt --rerank

Save CSV:
python src/eval.py --ckpt ./runs/best.pt --rerank --save_csv results.csv

---

## Inference

Single image:
python src/predict.py --ckpt runs/best.pt --image carimage.jpg

Directory:
python src/predict.py --ckpt runs/best.pt --dir ./samples

Optional:
python src/predict.py --ckpt runs/best.pt --image carimage.jpg --topk 5 --temperature 1.0

Debug:
python src/predict.py --ckpt runs/best.pt --image carimage.jpg --debug

---

## Recommended Workflow

# 1. Clone repo
git clone https://github.com/IsmailMehar/VehiReID-RAG.git
cd VehiReID-RAG

# 2. Setup environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add dataset
# Place CompCars under data/compcars/

# 5. Build index
python scripts/build_compcars_index.py

# 6. Train
python src/train.py

# 7. Evaluate
python src/eval.py --ckpt ./runs/best.pt

# 8. Inference
python src/predict.py --ckpt runs/best.pt --image carimage.jpg

---

## What NOT to Run

python src/datasets/compcars_dataset.py

(This file is internal, not a standalone script.)

---

## Configuration

Edit:
config/default.yaml

---

## Outputs

data/compcars/indexes/index.csv  
data/compcars/indexes/vocabs.json  
runs/best.pt  

---

## Troubleshooting

Missing index files:
python scripts/build_compcars_index.py

Missing checkpoint:
Check runs/ directory

Multi-GPU fails:
python src/train.py

Inference errors:
Check image path

---

## Limitations

- Focused on the CompCars dataset
- Limited generalisation without modification
- Multi-GPU requires a proper setup

---

## 📚 Citation

Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang.  
A Large-Scale Car Dataset for Fine-Grained Categorisation and Verification.  
CVPR, 2015.
