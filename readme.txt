# Video-Portrait-Tracker

## Project Overview

This project focuses on implementing and evaluating models for portrait tracking in videos. The key objective is to detect and track faces or other portrait features in a sequence of frames. We use a combination of bounding box techniques, Intersection over Union (IoU) evaluation, and advanced object detection models such as Faster R-CNN and YOLOv5.

## Table of Contents

- [Introduction](#introduction)
- [Key Techniques](#key-techniques)
- [Models Used](#models-used)
  - [Faster R-CNN](#faster-r-cnn)
  - [YOLOv5](#yolov5)
- [Training](#training)
- [Results](#results)
- [Future Work](#future-work)

## Introduction

Portrait tracking in videos is a challenging task due to the need for accurate detection and consistent tracking of moving subjects. This project implements several state-of-the-art algorithms to track and detect portraits in videos, using techniques like non-maximum suppression and Intersection over Union (IoU) for evaluating bounding boxes.

## Key Techniques

- **Bounding Box IoU**:  
  IoU is used to evaluate the accuracy of predicted bounding boxes. It is calculated as the area of overlap between the ground truth box and the predicted box, divided by the area of union. 

- **Non-Maximum Suppression**:  
  To eliminate redundant bounding boxes, non-maximum suppression is applied. The process involves selecting the bounding box with the highest confidence score and removing overlapping boxes whose IoU is greater than a threshold.

## Models Used

### Faster R-CNN

Faster R-CNN works in two stages:
1. **Region Proposal Network (RPN)**: Proposes potential bounding boxes for objects.
2. **CNN for Classification**: Classifies the proposed regions as either objects or background and refines the bounding box coordinates.

This method is known for its high accuracy but can be computationally intensive.

### YOLOv5

YOLO (You Only Look Once) splits the input image into a grid and predicts bounding boxes for each grid cell. YOLOv5, a faster version, is particularly efficient, making it suitable for real-time object detection tasks.

## Training

The model is trained using a custom dataset from **Roboflow**, which includes many anime characters. We used **Implicit Semantic Data Augmentation (ISDA)** to perform advanced data augmentation techniques, such as changing the background color and perspective of objects, to enhance training.

## Results

The models were evaluated using IoU and non-maximum suppression techniques. We observed strong performance from YOLOv5 in real-time video tracking, with Faster R-CNN excelling in precision but requiring more computational resources.

## Future Work

- Improve the model's real-time performance by optimizing inference speed.
- Apply more advanced augmentation techniques to generalize the model further.
- Explore the use of transformers for more accurate long-term tracking.
