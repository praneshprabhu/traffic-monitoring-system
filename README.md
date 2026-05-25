# 🚦 Real-Time Traffic Monitoring System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)](https://opencv.org)
[![YOLO](https://img.shields.io/badge/YOLOv8-Object%20Detection-red)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 📊 Project Overview

A real-time computer vision system that detects and counts vehicles using YOLOv8. Perfect for traffic analysis, smart cities, and parking management.

### Key Features
- ✅ **Real-time detection** at 30+ FPS
- ✅ **4 vehicle classes**: Cars, Motorcycles, Buses, Trucks
- ✅ **Line-crossing counting** with anti-double counting
- ✅ **Live statistics** display on screen
- ✅ **Fully offline** capability
- ✅ **Keyboard controls** (Q to quit, R to reset)

## 🎥 Demo

![Traffic Monitor Demo](docs/demo.gif)

*Real-time vehicle detection and counting in action*

## 🛠️ Technical Stack

| Technology | Purpose |
|------------|---------|
| Python 3.8+ | Core programming |
| YOLOv8 | Object detection model |
| OpenCV | Video processing & display |
| Ultralytics | YOLO implementation |

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (optional, works with video files)

### Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/traffic-monitoring-system.git
cd traffic-monitoring-system

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install ultralytics opencv-python

# Run application
python traffic_monitor.py