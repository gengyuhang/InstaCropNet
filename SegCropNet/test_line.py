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

def test():
    if os.path.exists('test_output') == False:
        os.mkdir('test_output')
    args = parse_args()
    img_path = args.img
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    model = LaneNet(arch=args.model_type)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
    dummy_input = torch.unsqueeze(dummy_input, dim=0)
    outputs = model(dummy_input)

    input = Image.open(img_path)
    input = input.resize((resize_width, resize_height))
    input = np.array(input)

    instance_pred = torch.squeeze(outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
    binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy() * 255

    cv2.imwrite(os.path.join('test_output', 'input.jpg'), input)
    cv2.imwrite(os.path.join('test_output', 'instance_output.jpg'), instance_pred.transpose((1, 2, 0)))
    cv2.imwrite(os.path.join('test_output', 'binary_output.jpg'), binary_pred)
    
    # postprocess_result = postprocessor.postprocess(
    #         binary_seg_result=binary_seg_image[0],
    #         instance_seg_result=instance_seg_image[0],
    #         source_image=image_vis,
    #         with_lane_fit=with_lane_fit,
    #         data_source='tusimple'
    #     )
    #     mask_image = postprocess_result['mask_image']
    #     if with_lane_fit:
    #         lane_params = postprocess_result['fit_params']
    #         LOG.info('Model have fitted {:d} lanes'.format(len(lane_params)))
    #         for i in range(len(lane_params)):
    #             LOG.info('Fitted 2-order lane {:d} curve param: {}'.format(i + 1, lane_params[i]))
    #
    #     for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
    #         instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
    #     embedding_image = np.array(instance_seg_image[0], np.uint8)
    #
    #     plt.figure('mask_image')
    #     plt.imshow(mask_image[:, :, (2, 1, 0)])
    #     plt.figure('src_image')
    #     plt.imshow(image_vis[:, :, (2, 1, 0)])
    #     plt.figure('instance_image')
    #     plt.imshow(embedding_image[:, :, (2, 1, 0)])
    #     plt.figure('binary_image')
    #     plt.imshow(binary_seg_image[0] * 255, cmap='gray')
    #     plt.show()

if __name__ == "__main__":
    test()

