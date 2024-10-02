# Video-Portrait-Tracker

## Project Overview

This project focuses on implementing and evaluating models for portrait tracking in videos. The key objective is to detect and track faces or other portrait features in a sequence of frames. We use a combination of bounding box techniques, Intersection over Union (IoU) evaluation, and advanced object detection models such as YOLOv5, fine-tuned for our specific dataset.

## Table of Contents

- [Introduction](#introduction)
- [Key Techniques](#key-techniques)
- [Models Used](#models-used)
  - [YOLOv5](#yolov5)
- [Training](#training)
- [Results](#results)
- [Future Work](#future-work)

## Introduction

Portrait tracking in videos is a challenging task due to the need for accurate detection and consistent tracking of moving subjects. This project implements state-of-the-art algorithms like YOLOv5 to track and detect portraits in videos, using techniques like non-maximum suppression and Intersection over Union (IoU) for evaluating bounding boxes.

## Key Techniques

- **Bounding Box IoU**:  
  IoU is used to evaluate the accuracy of predicted bounding boxes. It is calculated as the area of overlap between the ground truth box and the predicted box, divided by the area of union.

- **Non-Maximum Suppression**:  
  To eliminate redundant bounding boxes, non-maximum suppression is applied. The process involves selecting the bounding box with the highest confidence score and removing overlapping boxes whose IoU is greater than a threshold.

## Models Used

### YOLOv5

We used a pre-trained YOLOv5 model, which was fine-tuned on a custom anime dataset. YOLOv5 is known for its efficiency and is particularly suitable for real-time object detection tasks. 

The fine-tuning was performed on a dataset consisting of images from the anime *Yuan Shen*, and the final model was used to track portraits in a *Yuan Shen* trailer video.

## Training

The pre-trained YOLOv5 model was fine-tuned using the **Roboflow anime dataset**, which includes various anime character portraits. To enhance model performance, we applied **Implicit Semantic Data Augmentation (ISDA)**, which performs semantic transformations such as changing the background color and perspective of objects. This augmentation improved the training results, making the model more robust to different environments and conditions in the trailer video.

## Results

The final model was tested on a *Yuan Shen* trailer video, and it successfully tracked the portraits of characters throughout the video. The YOLOv5 model excelled in real-time portrait tracking, providing accurate and consistent results.

## Future Work

- Further optimize the model for even faster real-time performance.
- Expand the dataset with more diverse video content to improve generalization.
- Investigate the use of transformer-based models for improved tracking accuracy and efficiency.

