import cv2 as cv
import numpy as np
import torch

from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

from model.mobile import CustomMobileNetV2


class Worker:
    def __init__(self, img: bytes) -> None:
        self.img = img

    def get_info_list(self, results) -> list:
        # build list to store information
        cls_list = []
        xyxy_list = []

        for result in results:
            boxes = result.boxes

            # read information from boxes class
            clses = boxes.cls
            xyxyes = boxes.xyxy

        # convert tensor to numpy
        clses = clses.numpy()
        xyxyes = xyxyes.numpy()

        for cls in clses:
            cls_list.append(cls)

        for xyxy in xyxyes:
            xyxy_list.append(xyxy)
        
        # to recognize up or down channel, if reverse, swap
        for idx in range(len(cls_list) - 1):
            if cls_list[idx] > cls_list[idx+1]:
                cls_list[idx], cls_list[idx+1] = cls_list[idx+1], cls_list[idx]
                xyxy_list[idx], xyxy_list[idx+1] = xyxy_list[idx+1], xyxy_list[idx]

        if xyxy_list[0][1] > xyxy_list[1][1]:
            xyxy_list[0], xyxy_list[1] = xyxy_list[1], xyxy_list[0]

        return cls_list, xyxy_list

    def yolo_detect(self):
        """
        use yolov8 to detect roi and tube from img
        """

        # load weight and predict
        model = YOLO("weights/best.pt")

        # first detect
        results = model(self.img)
        cls_list, xyxy_list = self.get_info_list(results)

        # if tube is in left, adjust
        if xyxy_list[2][2] < xyxy_list[0][0]:
            self.img = cv.flip(self.img, -1)

            # second detect
            results = model(self.img)
            cls_list, xyxy_list = self.get_info_list(results)

        img_list = []

        for idx in range(len(xyxy_list)):
            # only crop roi region 
            if cls_list[idx] == 0:
                # read xmin, ymin, xmax, ymax
                xmin = int(xyxy_list[idx][0])
                ymin = int(xyxy_list[idx][1])
                xmax = int(xyxy_list[idx][2])
                ymax = int(xyxy_list[idx][3])

                # crop img
                img_crop = self.img[ymin:ymax, xmin:xmax]
                img_list.append(img_crop)

        return img_list[1], img_list[0]

    def predict(self, channel_down: bytes, channel_up: bytes) -> int:
        """
        use mobilenetv2 to predict channel class
        """

        # build model and load weight
        model = CustomMobileNetV2(num_classes=4)
        weight = torch.load("weights/MobileNet.pt")

        model.load_state_dict(weight)
        model.eval()

        # from numpy type to PIL Image
        channel_down = Image.fromarray(np.uint8(channel_down))
        channel_up = Image.fromarray(np.uint8(channel_up))


        # resize img to fit model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小为模型输入大小
            transforms.ToTensor(),  # 转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])

        channel_down = transform(channel_down).unsqueeze(0)
        channel_up = transform(channel_up).unsqueeze(0)

        # predict with model
        with torch.no_grad():
            output_down = model(channel_down)
            output_up = model(channel_up)

        _, predicted_down = output_down.max(1)
        _, predicted_up = output_up.max(1)

        channel_down_class = int(predicted_down.item())
        channel_up_class = int(predicted_up.item())

        return channel_down_class, channel_up_class
