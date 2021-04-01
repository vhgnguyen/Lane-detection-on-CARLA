#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
import numpy as np
import cv2
import config as cf

class PerspectiveTransform:
    
    def __init__(self, img_shape=(cf.IMG_HEIGHT, cf.IMG_WIDTH)):
        # parameters: perspective transform
        self.pp = {
            'st': 0.06, # src_top_width_scale, df: 0.06
            'sh': 0.55, # src_top_height_scale, df: 0.625
            'sb': -0.05, # src_bottom_width_offset, df: 0.07
            'dw': 0.25 # dst_width_scale, df: 0.35 for multiple lanes, 0.25 for one lane
        }
        self.img_shape = img_shape
        img_height = img_shape[0]
        img_width = img_shape[1]

        s1 = [img_width // 2 - img_width * self.pp['st'], img_height * self.pp['sh']]
        s2 = [img_width  // 2 + img_width * self.pp['st'], img_height * self.pp['sh']]
        s3 = [-img_width * self.pp['sb'], img_height]
        s4 = [img_width  + img_width * self.pp['sb'], img_height]
        src = np.float32([s1, s2, s3, s4])

        d1 = [img_width * self.pp['dw'], 0]
        d2 = [img_width * (1- self.pp['dw']), 0]
        d3 = [img_width * self.pp['dw'], img_height]
        d4 = [img_width * (1 - self.pp['dw']), img_height]
        dst = np.float32([d1, d2, d3, d4])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)
    
    def warpToBirdEyeView(self, img):
        return cv2.warpPerspective(img, self.M, (self.img_shape[1], self.img_shape[0]))

    def warpToCameraView(self, img):
        return cv2.warpPerspective(img, self.M_inv, (self.img_shape[1], self.img_shape[0]))
    
    def get(self):
        return self.M, self.M_inv, self.img_shape
    
    def getM(self):
        return self.M
    
    def getMinv(self):
        return self.M_inv


# ==============================================================================
# -- preprocessing -------------------------------------------------------------
# ==============================================================================


class ImageFilter:

    # Define a function that applies Sobel x or y, 
    # then takes an absolute value and applies a threshold.
    @staticmethod
    def abs_sobel_thresh(gray_img, orient='x', abs_thresh=(25,255), ksize=7):
        # Apply the following steps to img
        # 1) Convert to grayscale === or LAB L channel
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        dx = int(orient=='x')
        dy = int(orient=='y')
        sobel = cv2.Sobel(gray_img, cv2.CV_64F, dx, dy,ksize=ksize)
        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude 
                # is > thresh_min and < thresh_max
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= abs_thresh[0]) & (scaled_sobel <= abs_thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        binary_output = sxbinary # Remove this line
        return binary_output

    # Define a function that applies Sobel x and y, 
    # then computes the magnitude of the gradient
    # and applies a threshold
    @staticmethod
    def mag_thresh(gray_img, ksize=7, mag_thresh=(25, 255)):
        # Apply the following steps to img
        # 1) Convert to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=ksize)
        # 3) Calculate the magnitude 
        mag_sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
        # 5) Create a binary mask where mag thresholds are met
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        binary_output = np.copy(sxbinary) 
        return binary_output

    # Define a function that applies Sobel x and y, 
    # then computes the direction of the gradient
    # and applies a threshold.
    @staticmethod
    def dir_thresh(gray_img, ksize=7, dir_thresh=(0, 0.09)):    
        # Apply the following steps to img
        # 1) Convert to grayscale
        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=ksize)
        # 3) Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
        grad_dir = np.arctan2(abs_sobely, abs_sobelx)
        # 5) Create a binary mask where direction thresholds are met
        binary_output =  np.zeros_like(grad_dir)
        binary_output[(grad_dir >= dir_thresh[0]) & (grad_dir <= dir_thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return binary_output

    # Define a function that thresholds the S-channel of HLS
    # Use exclusive lower bound (>) and inclusive upper (<=)
    @staticmethod
    def hls_sthresh(hls, hls_thresh=(125, 255)):
        # 1) Convert to HLS color space
        # hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        # 2) Apply a threshold to the S channel
        binary_output = np.zeros_like(hls[:,:,2])
        binary_output[(hls[:,:,2] > hls_thresh[0]) & (hls[:,:,2] <= hls_thresh[1])] = 1
        # 3) Return a binary image of threshold result
        return binary_output

    # Define a function that thresholds the L-channel of HLS
    # Use exclusive lower bound (>) and inclusive upper (<=)
    @staticmethod
    def hls_lthresh(hls, thresh=(220, 255)):
        # 1) Convert to HLS color space
        # hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hls_l = hls[:,:,1]
        hls_l = hls_l*(255/np.max(hls_l))
        # 2) Apply a threshold to the L channel
        binary_output = np.zeros_like(hls_l)
        binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
        # 3) Return a binary image of threshold result
        return binary_output

    @staticmethod
    def applyFilter(img):
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        s_chnl = hls_img[:,:,2]
        l_chnl = hls_img[:,:,1]

        ds_x = ImageFilter.abs_sobel_thresh(gray_img, orient='x', ksize=5, abs_thresh=(25,255))
        ds_y = ImageFilter.abs_sobel_thresh(gray_img, orient='y', ksize=5, abs_thresh=(25,200))

        # s_b = np.zeros_like(s_chnl)
        # s_thresh = cv2.inRange(s_chnl.astype('uint8'),80,250)
        # s_b[(s_thresh==255)] = 1

        l_b = np.zeros_like(l_chnl)
        l_thresh = cv2.inRange(l_chnl.astype('uint8'),220,250)
        l_b[(l_thresh==255)] = 1

        combined_b = np.zeros_like(ds_x)
        combined_b[(l_b == 1) | (ds_x == 1) | (ds_y == 1)] = 1

        return combined_b