#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         image_A_image
# Description:
# Author:       Ming_King
# Date:         2024/3/7
# -------------------------------------------------------------------------------
import cv2
import numpy as np

image1 = cv2.imread(r'E:\A_trans\results\AH\3\Deeplabv3+\249_input.png')
image2 = cv2.imread(r'E:\A_trans\results\AH\3\Deeplabv3+\249_binary_output.png')
# image2 = cv2.morphologyEx(image2, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
img = cv2.addWeighted(image1, 1, image2, 0.5, 0)

cv2.imshow('Mask', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Save the resulting image
cv2.imwrite('change_add_2.jpg', img)