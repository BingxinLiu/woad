#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 01:27:15 2021

@author: hliu
"""

import numpy as np
import cv2
#import argparse
import sys
from collections import Counter

def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
    # SGBM Parameters -----------------
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    #displ = np.int16(displ)
    #dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
    org_filteredImg = np.copy(filteredImg)
    #min_max of unnormalized Img
#    print("Max and min:")
#    print(filteredImg.max())
#    print(filteredImg.min())
    
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    #n_filteredImg = cv2.normalize(src=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    
#    print("OOOOOO Max and min:")
#    print(org_filteredImg.max())
#    print(org_filteredImg.min())
    
#    cv2.imshow('displ', displ)
#    cv2.imshow('dispr', dispr)

    return filteredImg, org_filteredImg

def convex_hull(gray):
    
    ret, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    # 寻找图像中的轮廓
    contours, hierarchy = cv2.findContours(thresh, 2, 1)
    
    # 寻找物体的凸包并绘制凸包的轮廓
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        length = len(hull)
        # 如果凸包点集中的点个数大于5
        if length > 2:
            # 绘制图像凸包的轮廓
            for i in range(length):
                cv2.line(gray, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,0,255), 2)
    
    cv2.imshow('Concex Hull', gray)
    
def min_max(disparity_image, org_image):
    
    #delete lines in the last
    disparity_image = disparity_image[:-20][:]
    
    disp_max = disparity_image.max(axis = 0)
    
    max_array = np.ones([30,640]) * disp_max
    max_num_array = np.where(disparity_image==disp_max)
    max_num_array1 = max_num_array[1]
    C = Counter(max_num_array1)
    freq = [C[i] for i in range(640)]
    #print(freq)
    freq_array = np.ones([30,640]) * np.array(freq)
    #freq_array = freq_array * max_array
    
    uint_img = np.array(max_array).astype('uint8')
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    uint_freq = np.array(freq_array).astype('uint8')
    grayFreq = cv2.cvtColor(uint_freq, cv2.COLOR_GRAY2BGR)
    
    #max_array = cv2.normalize(src=max_array, dst=max_array, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX，dtype=cv2.CV_8UC1);
    cv2.imshow('gray', grayImage)
    #cv2.imshow('Freq', grayFreq)
    
    org_max = org_image.max(axis = 0)
    
    disp_area = np.zeros([1,5]) + (-16)
    item_area = np.zeros([1,5])
    distance = np.zeros([1,5])
    
    #2m???
    n = 2.6 * 60 / 300
    
    for i in range(5):
        disp_area[0][i], distance[0][i] = topk(org_max,i)
        if distance[0][i] < n:
            item_area[0][i] = area(org_max,i,n)
    
    #print("Org max", org_image.max())
    print("distance",distance)
    print("disp_area",disp_area)
    print("item_area",item_area)
    
    return distance, disp_area, item_area
    
    
    #find 0: num of 0, which is black in the disp map = 80
    
#    disp_sum = np.sum(disparity_image,axis=0)
#    zero_C = Counter(disp_sum)
#    zero_num = zero_C[0]
#    print(zero_num)

def topk(disp_max, i):
    
    top_k = 20
    #print(i)
    disp_i = disp_max[80 + 112 * i : 192 + 112 * i]
    top_k_idx=disp_i.argsort()[::-1][0:top_k]
    top_k_value = disp_i[top_k_idx]
    #print(top_k_value)
    #print(disp_i.max())
    top_k_dis = (2.6 * 60) / top_k_value
    
    return top_k_value.mean(), top_k_dis.mean()

def area(disp_max,i,n):
    
    item = disp_max[80 + 112 * i : 192 + 112 * i]
    idex = np.argwhere(item > ((2.6 * 60) / n))
    item = item[idex]
    item_sum = np.sum(item)
    
    return item_sum
    
    


if __name__ == '__main__':

    # is camera stream or video
    if True:
        #cap_left = cv2.VideoCapture(args.left_source, cv2.CAP_V4L2)
        #cap_right = cv2.VideoCapture(args.right_source, cv2.CAP_V4L2)
        dispW = 640
        dispH = 480
        flip = 0
        
        cam1Set = 'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
        cam2Set = 'nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
        
        cap_right = cv2.VideoCapture(cam1Set)
        cap_left = cv2.VideoCapture(cam2Set)

    #print("Start")
    #K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(args.calibration_file)  # Get cams params
#    K1 = np.array([[5.717662471089529e+02, 0, 3.599894403338598e+02],[0, 7.636306316305219e+02, 2.947943548027684e+02],[0, 0, 1]])
#    D1 = np.array([[-0.284128304263526, 0.603976907145943, 0.002650751621832, 6.311318302912327e-04, 0.00000]])
#    K2 = np.array([[5.679819209188780e+02, 0, 3.452859895943097e+02],[0, 7.609519703287510e+02, 2.895787685416111e+02],[0, 0, 1]])
#    D2 = np.array([[-0.236473331555064, 0.319212543138987, 0.002170239381114, -0.002597546865672, 0.00000]])
#    R = np.array([[1, -0.001208891549226, -0.002628955911107],[0.001199053991384, 1, -0.003740364031551],[0.002633457325662, 0.003737196112728, 1]])
#    T = np.array([-60.513327359166034, -0.037255253186699, -2.189518833850237])
    
    K1 = np.array([[5.679286815218036e+02, 0, 3.427315334157044e+02],\
                   [0, 7.569521550160922e+02, 2.749032672507641e+02],\
                   [0, 0, 1]])
    D1 = np.array([[-0.247898597613495, 0.963612376455093, \
                    0.005007073878467, 0.003544239236512, 0.00000]])
    K2 = np.array([[5.665439587514537e+02, 0, 3.300422688805023e+02],\
                   [0, 7.557038654043611e+02, 2.679839339287851e+02],\
                   [0, 0, 1]])
    D2 = np.array([[-0.242511849444114, 0.980331254691578,\
                    0.002963361851048, 0.003507404137556, 0.00000]])
    R = np.array([[0.999809488415120, -3.803081241759167e-04, 0.019515179753873],\
                  [4.462727193334061e-04, 1, -0.003375924839205],\
                  [-0.019513782712618, 0.003383990778750, 0.999803861210115]])
    T = np.array([-58.953856363020520, 0.403802936505059, -1.051816198390031])
    
    rotation1, rotation2, pose1, pose2 = cv2.stereoRectify(cameraMatrix1 = K1,
                          distCoeffs1 = D1,
                          cameraMatrix2 = K2,
                          distCoeffs2 = D2,
                          imageSize=(dispW, dispH),
                          R = R,
                          T = T,
                          #flags=cv2.CALIB_ZERO_DISPARITY,
                          #newImageSize=(dispW, dispH)
                          )[0:4]
    #print("Retify coefficients calculated")
    

    if not cap_left.isOpened() and not cap_right.isOpened():  # If we can't get images from both sources, error
        print("Can't opened the streams!")
        sys.exit(-9)

    # Change the resolution in need
#    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
#    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float
#
#    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
#    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float

    while True:  # Loop until 'q' pressed or stream ends
        # Grab&retreive for sync images
        if not (cap_left.grab() and cap_right.grab()):
            print("No more frames")
            break

        _, leftFrame = cap_left.retrieve()
        _, rightFrame = cap_right.retrieve()
        height, width, channel = leftFrame.shape  # We will use the shape for remap
        #print("Retrieve")

        # Undistortion and Rectification part!

        leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1,
                                               D1,
                                               rotation1, pose1, (dispW, dispH), cv2.CV_32FC1)
        left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2,
                                               D2,
                                               rotation2, pose2, (dispW, dispH), cv2.CV_32FC1)
        right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        
        #print("Rectified!!!")

        # We need grayscale for disparity map.
        gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        #print("ToGray.")

        disparity_image, org_image = depth_map(gray_left, gray_right)  # Get the disparity map
        #print("Get Disp Map")
        #print(type(disparity_image))
        #print(np.shape(disparity_image))
        
        depth_array = (2.6 * 60) / org_image

        # Show the images
        #cv2.imshow('left(R)', left_rectified )
        #cv2.imshow('right(R)', right_rectified)
        cv2.imshow('Disparity', disparity_image)
        cv2.imshow('xnom Disparity', org_image)
        #convex_hull(disparity_image)
        
        distance, disp_area, item_area = min_max(disparity_image, org_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Get key to stop stream. Press q for exit
            break

    # Release the sources.
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
