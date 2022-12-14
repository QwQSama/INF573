from os.path import exists
import argparse
import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from ISDA import FullLayer
from faster_rcnn import fasterrcnn_resnet50_fpn

parser = argparse.ArgumentParser()

parser.add_argument('--input_path', type=str, default='Genshin.mp4')
parser.add_argument('--in_channels', type=int, default=12544)
parser.add_argument('--representation_size', type=int, default=1024)
parser.add_argument('--model', type=str, default='fasterrcnn_resnet50_anime.pth')
parser.add_argument('--fc', type=str, default='fasterrcnn_fc.pth')
parser.add_argument('--output_path', type=str, default='Genshin_FasterRCNN/')

args = parser.parse_args()

model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)
if exists(args.model):
    model.load_state_dict(torch.load(args.model))

fc = FullLayer(args.in_channels, args.representation_size)
if exists(args.fc):
    fc.load_state_dict(torch.load(args.fc))

model = model.cuda()
fc = fc.cuda()
model.eval()
fc.eval()

data_transform = transforms.ToTensor()

input_video = cv2.VideoCapture(args.input_path)
fps = input_video.get(cv2.CAP_PROP_FPS)
size = (int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

videoWriter = cv2.VideoWriter(args.output_path,
                              cv2.VideoWriter_fourcc("M", "J", "P", "G"),  # Encoder
                              fps,
                              size)

success, frame = input_video.read()
while success:  # Loop until finish
    img = data_transform(frame)
    img = torch.unsqueeze(img, dim=0)
    img = img.cuda()

    output = model(img)
    output = fc(output)

    # Extract results
    boxes = output[0]['boxes'].data.cpu().numpy()
    scores = output[0]['scores'].data.cpu().numpy()

    img = img[0].cpu().numpy().transpose(1, 2, 0)
    print(len(img))
    print(len(img[0]))
    img *= 255
    img = np.ascontiguousarray(img, dtype=np.uint8)
    # Draw bounding boxes
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
