---
title: Prenatal Vision DL
emoji: ⚪
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Prenatal Vision DL Server
This is the Deep Learning inference server for the Prenatal Vision project. It hosts multiple YOLO11-based models for CRL and NT detection and measurement in ultrasound images.

## Deployment Features
- Azure Downloader: Automatically pulls model weights from Azure Blob Storage.
- Orphan Push: History-less deployment for high performance on Hugging Face.
- Dockerized: Runs with Gunicorn and Flask for scalable inference.