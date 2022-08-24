#!/usr/bin/python

import cv2
import cv2.aruco as aruco 
import numpy as np 
import os
import csv 
import time
import scipy.io
import imutils
# Reads in a video file taken with my iPhone, and runs it thorugh frame-by-frame, tracking a SINGLE aruco marker with specific id in it, and saving the transformation from camera to aruco marker as a .csv
# Robert Baines, 18 Nov, 2022 

# script formatted by following: https://www.youtube.com/watch?v=v5a7pKSOJd8 
# and got the pose and coordinate axis drawing thing from: https://github.com/njanirudh/Aruco_Tracker/blob/master/aruco_tracker.py



def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #tell the format of aruco marker to look for: 
    arucoDict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray,arucoDict,parameters=arucoParam)

    if draw: 
        aruco.drawDetectedMarkers(img, bboxs)
    return [bboxs, ids]




def main():

    # From kevin's calibration code, using (Robert's iPhone): 
    #Camera Matrix:
    mtx=np.array([[1.78506701e+03,0.00000000e+00,9.87216048e+02],[0.00000000e+00,1.79787094e+03,5.23113084e+02],[0.00000000e+00,0.00000000e+00,1.00000000e+00]])
    # Distortion Parameters:
    dist=np.array([2.26823628e-01,  -1.09492739e+00,   2.13016860e-03,  -3.70317149e-04, 1.86184423e+00])

    # read in the video
    cap = cv2.VideoCapture('/Users/robertbaines/Desktop/NCCR_exchange/light_sensor_localFolder/aruco_test/ground_truth_video.MOV') # note: https://stackoverflow.com/questions/44380432/opencv-video-looks-good-but-frames-are-rotated-90-degrees ... open cv doesn't record video orientation from mobile, so that's why my video read in rotated by 90 degrees from the orientation I take them in. 
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # get frame number last the accurate way... #cap.get(cv2.CAP_PROP_FRAME_COUNT) # CAP isn't very good... actually is several off from actual frames, which I manually counted as 1567 in loop.
    success=True
    count=0
    print("Calculating frames in video, this could take a minute...")
    while success:
        success,image=cap.read()
        if success==False:
            break
        count=count+1

    last_frame_num = count
    print("There are ",last_frame_num, " frames in the video")

    # read in the video again. 
    cap.release()
    cap = cv2.VideoCapture('/Users/robertbaines/Desktop/NCCR_exchange/light_sensor_localFolder/aruco_test/ground_truth_video.MOV') # note: https://stackoverflow.com/questions/44380432/opencv-video-looks-good-but-frames-are-rotated-90-degrees ... open cv doesn't record video orientation from mobile, so that's why my video read in rotated by 90 degrees from the orientation I take them in. 
    
    #initialize storage matrices 
    rvec_save = []
    tvec_save = []
    t_mat_save = []
    inv_t_mat_save = []
    time_stamp = []
    r_cam_global_save = []
    fram_cnt = 1
    cnt = 1 
    while cap.isOpened():

        ret,frame = cap.read()
        #frame = imutils.rotate(frame, 90)  # roate image correctly (to prevent cutting off by open cvs function:https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/)

        fram_cnt = fram_cnt + 1 
        print(fram_cnt)
        found_bboxs, found_ids = findArucoMarkers(frame)

        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(found_bboxs, 0.05, mtx, dist)  # estimate the pose of the aruco marker

        # check if the ids list is not empty; if no check is added the code will crash
        if np.all(found_ids != None):

            # create transformation matrix from marker to camera (camera is glocal coordinate system, since stationary) using rvec and tvec: https://stackoverflow.com/questions/67416476/transformation-matrix-from-estimateposesinglemarkers
            rotation_matrix = np.array(cv2.Rodrigues(rvec)[0])
            transformation_matrix = np.zeros([4, 4])   
            transformation_matrix[0:3, 0:3] = rotation_matrix         
            transformation_matrix[0:3, 3] = tvec
            transformation_matrix[3, 3] = 1

            #  invert this matrix to get the transformation from camera to marker:
            inv_transformation_matrix = np.linalg.inv(transformation_matrix)

            #calculate the transform between global and marker (we're flipping camera 180 about X-axis, to align y with "up" screen direction, and Z with "out" screen direction)
            r_cam_global = np.matmul( rotation_matrix , np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]]) )

            # draw axis for the aruco markers.
            #The marker coordinate system that is assumed by this function is placed at the center of the marker with the Z axis pointing out
            # Axis-color correspondences are X: red, Y: green, Z: blue.
            aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1)

            #print(r_cam_global)

        else: # if no aruco marker found, save parameters as 0's just to keep consistent time stamp with each fframe. In post, we can remove these zeros with previous frame's orientatino or something. 
            rvec = np.zeros([1,3])
            tvec = np.zeros([1,3])
            inv_transformation_matrix = np.zeros([4, 4])
            transformation_matrix = np.zeros([4,4])
            r_cam_global = np.zeros([3, 3])

        rvec = np.reshape(rvec, [1,3])

        # save the transformation matrix and other stuff. 
        rvec_save.append(rvec)
        tvec_save.append(tvec)
        inv_t_mat_save.append(inv_transformation_matrix)
        t_mat_save.append(transformation_matrix)
        r_cam_global_save.append(r_cam_global)

        time_stamp.append( (1/frame_rate) * fram_cnt) 
        
        print(r_cam_global)

       # display the resulting frame
        frame = cv2.resize(frame,(600,400))
        cv2.imshow('frame',frame)

        # break loop before last frame is reached, else script will crash and csv won't be saved. 
        if last_frame_num == fram_cnt:
            break 

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    scipy.io.savemat('/Users/robertbaines/Desktop/NCCR_exchange/light_sensor_localFolder/aruco_test/timeStamp2.mat', mdict={'arr': time_stamp})    
    scipy.io.savemat('/Users/robertbaines/Desktop/NCCR_exchange/light_sensor_localFolder/aruco_test/rcamglobal2.mat', mdict={'arr': r_cam_global_save})    
    scipy.io.savemat('/Users/robertbaines/Desktop/NCCR_exchange/light_sensor_localFolder/aruco_test/tvec2.mat', mdict={'arr': tvec_save})    


    cap.release()
    cv2.destroyAllWindows() # destroy all opened windows


if __name__ == "__main__": 
    main()


