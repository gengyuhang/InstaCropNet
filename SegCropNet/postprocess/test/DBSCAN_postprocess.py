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
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = args.model
    model = SegCropNet(arch=args.model_type)
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

# def init_args():
#     """
#
#     :return:
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
#     parser.add_argument('--weights_path', type=str, help='The model weights path')
#     parser.add_argument('--with_lane_fit', type=args_str2bool, help='If need to do lane fit', default=True)
#
#     return parser.parse_args()
#
#
# def args_str2bool(arg_value):
#     """
#
#     :param arg_value:
#     :return:
#     """
#     if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#
#     elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Unsupported value encountered.')
#
#
# def minmax_scale(input_arr):
#     """
#
#     :param input_arr:
#     :return:
#     """
#     min_val = np.min(input_arr)
#     max_val = np.max(input_arr)
#
#     output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)
#
#     return output_arr
#
#
# def test_SegCropNet(image_path, weights_path, with_lane_fit=True):
#     """
#
#     :param image_path:
#     :param weights_path:
#     :param with_lane_fit:
#     :return:
#     """
#     assert ops.exists(image_path), '{:s} not exist'.format(image_path)
#
#     LOG.info('Start reading image and preprocessing')
#     t_start = time.time()
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     image_vis = image
#     image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
#     image = image / 127.5 - 1.0
#     LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))
#
#     input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
#
#     net = lanenet.LaneNet(phase='test', cfg=CFG)
#     binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')
#
#     postprocessor = SegCropNet_postprocess.LaneNetPostProcessor(cfg=CFG)
#
#     # Set sess configuration
#     sess_config = tf.ConfigProto()
#     sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
#     sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
#     sess_config.gpu_options.allocator_type = 'BFC'
#
#     sess = tf.Session(config=sess_config)
#
#     # define moving average version of the learned variables for eval
#     with tf.variable_scope(name_or_scope='moving_avg'):
#         variable_averages = tf.train.ExponentialMovingAverage(
#             CFG.SOLVER.MOVING_AVE_DECAY)
#         variables_to_restore = variable_averages.variables_to_restore()
#
#     # define saver
#     saver = tf.train.Saver(variables_to_restore)
#
#     with sess.as_default():
#         saver.restore(sess=sess, save_path=weights_path)
#
#         t_start = time.time()
#         loop_times = 500
#         for i in range(loop_times):
#             binary_seg_image, instance_seg_image = sess.run(
#                 [binary_seg_ret, instance_seg_ret],
#                 feed_dict={input_tensor: [image]}
#             )
#         t_cost = time.time() - t_start
#         t_cost /= loop_times
#         LOG.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))
#
#         postprocess_result = postprocessor.postprocess(
#             binary_seg_result=binary_seg_image[0],
#             instance_seg_result=instance_seg_image[0],
#             source_image=image_vis,
#             with_lane_fit=with_lane_fit,
#             data_source='tusimple'
#         )
#         mask_image = postprocess_result['mask_image']
#         if with_lane_fit:
#             lane_params = postprocess_result['fit_params']
#             LOG.info('Model have fitted {:d} lanes'.format(len(lane_params)))
#             for i in range(len(lane_params)):
#                 LOG.info('Fitted 2-order lane {:d} curve param: {}'.format(i + 1, lane_params[i]))
#
#         for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
#             instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
#         embedding_image = np.array(instance_seg_image[0], np.uint8)
#
#         plt.figure('mask_image')
#         plt.imshow(mask_image[:, :, (2, 1, 0)])
#         plt.figure('src_image')
#         plt.imshow(image_vis[:, :, (2, 1, 0)])
#         plt.figure('instance_image')
#         plt.imshow(embedding_image[:, :, (2, 1, 0)])
#         plt.figure('binary_image')
#         plt.imshow(binary_seg_image[0] * 255, cmap='gray')
#         plt.show()
#
#     sess.close()
#
#     return
#
#
# if __name__ == '__main__':
#     """
#     test code
#     """
#     # init args
#     args = init_args()
#
#     test_lanenet(args.image_path, args.weights_path, with_lane_fit=args.with_lane_fit)