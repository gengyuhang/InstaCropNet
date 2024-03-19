import time
import os
import sys

import torch
from dataloader.transformers import Rescale
from model.SegCropNet.SegCropNet import SegCropNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from model.utils.cli_helper_test import parse_args
import numpy as np
from PIL import Image
import pandas as pd
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img

def test(if_single_img=True):
    if os.path.exists('test_output') == False:
        os.mkdir('test_output')
    args = parse_args()
    img_path = args.img
    file_path = args.Mutil_img
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    model = SegCropNet(arch=args.model_type)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)


    if not if_single_img:
        # Get a list of all image files in the input folder
        image_files = [f for f in os.listdir(file_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        for image_file in image_files:
            # Read the image
            img_path = os.path.join(file_path, image_file)
            # 提取文件名（不包含后缀）
            file_name = os.path.splitext(os.path.basename(image_file))[0]

            dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
            dummy_input = torch.unsqueeze(dummy_input, dim=0)
            outputs = model(dummy_input)

            input = Image.open(img_path)
            input = input.resize((resize_width, resize_height))
            input = np.array(input)

            instance_pred = torch.squeeze(outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
            binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy() * 255


            cv2.imwrite(os.path.join('test_output', f"{file_name}_input.jpg"), input)
            cv2.imwrite(os.path.join('test_output', f"{file_name}_instance_output.jpg"), instance_pred.transpose((1, 2, 0)))
            cv2.imwrite(os.path.join('test_output', f"{file_name}_binary_output.jpg"), binary_pred)
    else:
        dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
        dummy_input = torch.unsqueeze(dummy_input, dim=0)
        outputs = model(dummy_input)

        input = Image.open(img_path)
        input = input.resize((resize_width, resize_height))
        input = np.array(input)

        instance_pred = torch.squeeze(outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
        binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy() * 255

        file_name1 = './test_out/input.jpg'
        file_name2 = './test_out/instance_output.jpg'
        file_name3 = './test_out/binary_output.jpg"'

        cv2.imwrite(os.path.join('test_output', 'input.jpg'), input)
        cv2.imwrite(os.path.join('test_output', 'instance_output.jpg'), instance_pred.transpose((1, 2, 0)))
        cv2.imwrite(os.path.join('test_output', 'binary_output.jpg'), binary_pred)

    return file_name1,file_name2,file_name3


if __name__ == "__main__":
    test(False)