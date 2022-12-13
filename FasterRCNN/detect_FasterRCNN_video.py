# coding:utf8
from os.path import exists
import sys
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import cv2
import numpy as np

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)
model.load_state_dict(torch.load('fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'))
model = model.cuda()
model.eval()

data_transform = transforms.ToTensor()

input_video = cv2.VideoCapture("Thorin_Input.mp4")
fps = input_video.get(cv2.CAP_PROP_FPS)
size = (
    int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

videoWriter = cv2.VideoWriter(
    "output_video.mp4",
    cv2.VideoWriter_fourcc("M", "J", "P", "G"),  # 编码器
    fps,
    size
)

'''
编码器常用的几种：
cv2.VideoWriter_fourcc("M", "J", "P", "G")  
#  视频MP4

cv2.VideoWriter_fourcc("I", "4", "2", "0") 
#    压缩的yuv颜色编码器，4:2:0色彩度子采样 兼容性好，产生很大的视频 avi

cv2.VideoWriter_fourcc("P", "I", "M", "I")
#    采用mpeg-1编码，文件为avi

cv2.VideoWriter_fourcc("X", "V", "T", "D")
#    采用mpeg-4编码，得到视频大小平均 拓展名avi

cv2.VideoWriter_fourcc("T", "H", "E", "O")
#    Ogg Vorbis， 拓展名为ogv

cv2.VideoWriter_fourcc("F", "L", "V", "1")
#    FLASH视频，拓展名为.flv
'''

success, frame = input_video.read()
while success:  # 循环直到没有帧了
    img = data_transform(frame)
    img = torch.unsqueeze(img, dim=0)
    img = img.cuda()

    output = model(img)
    # 提取检测结果
    boxes = output[0]['boxes'].data.cpu().numpy()
    scores = output[0]['scores'].data.cpu().numpy()

    img = img[0].cpu().numpy().transpose(1, 2, 0)
    print(len(img))
    print(len(img[0]))
    img *= 255
    img = np.ascontiguousarray(img, dtype=np.uint8)
    # 绘制检测结果
    for i in range(len(boxes)):
        if scores[i] > 0.8:
            print(scores[i])
            box = boxes[i]
            x1, y1, x2, y2 = box
            print(box)
            cv2.rectangle(frame, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(255, 0, 0), thickness=2)

    videoWriter.write(frame)
    success, frame = input_video.read()
videoWriter.release()