from os.path import exists
import sys
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import cv2
import numpy as np

# load pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)
model.load_state_dict(torch.load('fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'))
model = model.cuda()
model.eval()

# read image and preprocess
images = []
# start = sys.argv[1]
# start = int(start)
for i in range(1, 450):
    images.append('Image' + str(i) + '.jpg')
for frame in range(len(images)):
    print(frame, '/449')
    img = cv2.imread('Eastwood/' + images[frame])
    # define preprocess
    data_transform = transforms.ToTensor()
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    img = img.cuda()
    # detect
    output = model(img)
    # extract result
    boxes = output[0]['boxes'].data.cpu().numpy()
    scores = output[0]['scores'].data.cpu().numpy()
    # img = img * 255
    # img = img[0].to(torch.uint8).numpy()
    img = img[0].cpu().numpy().transpose(1, 2, 0)
    print(len(img))
    print(len(img[0]))
    img *= 255
    img = np.ascontiguousarray(img, dtype=np.uint8)
    # draw bounding box
    for i in range(len(boxes)):
        if scores[i] > 0.8:
            print(scores[i])
            box = boxes[i]
            x1, y1, x2, y2 = box
            print(box)
            cv2.rectangle(img, pt1=(int(x1), int(y1)), pt2=(int(x2), int(y2)), color=(255, 0, 0), thickness=2)

    cv2.rectangle(img, pt1=(0, 0), pt2=(180, 80), color=(0, 0, 0), thickness=-1)
    # save result
    # cv2.imshow('img', img)
    img_path = str(frame + 1)
    if len(img_path) == 1:
        img_path = '00' + img_path + '.jpg'
    elif len(img_path) == 2:
        img_path = '0' + img_path + '.jpg'
    else:
        img_path = img_path + '.jpg'

    print(img_path)
    cv2.imwrite('./Eastwood_output/' + img_path, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
