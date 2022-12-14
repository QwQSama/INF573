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

parser.add_argument('--input_path', type=str, default='./Genshin/')
parser.add_argument('--picture_nums', type=int, default=369)
parser.add_argument('--in_channels', type=int, default=12544)
parser.add_argument('--representation_size', type=int, default=1024)
parser.add_argument('--model', type=str, default='./fasterrcnn_resnet50_anime.pth')
parser.add_argument('--fc', type=str, default='./fasterrcnn_fc.pth')
parser.add_argument('--output_path', type=str, default='./Genshin_output/')

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

images = []

for i in range(1, args.picture_nums):
    images.append(args.input_path + str(i) + '.jpg')
for frame in range(len(images)):
    print(frame + 1, args.picture_nums)
    img = cv2.imread(args.input_path + images[frame])
    # Preprocess
    data_transform = transforms.ToTensor()
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.cuda()
    # Detect
    output = model(img)
    # Extract results
    boxes = output[0]['boxes'].data.cpu().numpy()
    scores = output[0]['scores'].data.cpu().numpy()
    
    img = img[0].cpu().numpy().transpose(1, 2, 0)
    
    img *= 255
    img = np.ascontiguousarray(img, dtype=np.uint8)
    # Draw bounding boxes
    for i in range(len(boxes)):
        if scores[i] > 0.8:
            print(scores[i])
            box = boxes[i]
            x1, y1, x2, y2 = box
            print(box)
            cv2.rectangle(img, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(255, 0, 0), thickness=20)

    img_path = str(frame + 1)
    if len(img_path) == 1:
        img_path = '00' + img_path + '.jpg'
    elif len(img_path) == 2:
        img_path = '0' + img_path + '.jpg'
    else:
        img_path = img_path + '.jpg'

    print(img_path)
    cv2.imwrite(args.output_path + img_path, img)
