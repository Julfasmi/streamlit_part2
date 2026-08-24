# 👁️ Like & Comment Detection with YOLOv8 + Streamlit

> An interactive computer vision application for detecting **Like** and **Comment** elements in social media screenshots using a custom-trained YOLOv8 object detection model.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-111111?style=flat)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)

---

## 📌 Overview

This project demonstrates the deployment of a custom **object detection model** through an interactive web application built with **Streamlit**.

The application is designed to detect **Like** and **Comment** elements from screenshots captured from social media platforms such as:

- Facebook
- YouTube
- TikTok

Users can upload an image through the web interface, and the application automatically performs object detection using a custom-trained **YOLOv8s model**.

The detected objects are then displayed with bounding boxes directly in the application.

---

## 🎯 Objective

The main objective of this project is to demonstrate how a trained computer vision model can be transformed into an interactive application that can be used by non-technical users.

The project combines:

**Computer Vision + Deep Learning + Model Inference + Web Application**

rather than stopping at model training and evaluation.

---

## 🔍 How It Works

The application follows this workflow:

```text
User Uploads Image
        │
        ▼
Image Preprocessing
        │
        ▼
YOLOv8 Object Detection Model
        │
        ▼
Model Inference
        │
        ▼
Bounding Box Detection
        │
        ▼
Annotated Image
        │
        ▼
Display Detection Results
