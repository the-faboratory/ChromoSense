#!/usr/bin/python

# This script uses the trained NN in matlab to predict the rotation matrix from RGB values. 

# importing matlab structure into python tutorial followed here: 
# https://docs.scipy.org/doc/scipy/reference/tutorial/io.html

import scipy.io as sio 
import cv2.aruco as aruco 
import numpy as np 
import os
import serial

def my_ann_evaluation(my_ann_structure, input): 


    # extract fields as variables
    ymax = my_ann_structure[0,0].input_ymax;
    ymin = my_ann_structure[0,0].input_ymin;
    xmax = my_ann_structure[0,0].input_xmax;
    xmin = my_ann_structure[0,0].input_xmin;

    input_preprocessed = (ymax-ymin) * (input-xmin) / (xmax-xmin) + ymin;

    # Pass it through the ANN matrix multiplication
    y1 = np.tanh(   np.matmul( my_ann_structure[0,0].IW ,input_preprocessed) + my_ann_structure[0,0].b1);
    y2 = np.matmul( my_ann_structure[0,0].LW , y1)  + my_ann_structure[0,0].b2;

    ymax = my_ann_structure[0,0].output_ymax;
    ymin = my_ann_structure[0,0].output_ymin;
    xmax = my_ann_structure[0,0].output_xmax;
    xmin = my_ann_structure[0,0].output_xmin;

    res = (y2-ymin) * (xmax-xmin) /(ymax-ymin) + xmin;

    #return the result of the neural network: 
    return res 





def main():

    ser = serial.Serial("/dev/cu.usbmodem14201", 57600)

    # first, load our neural network structure that we exported from matlab. 
    mat_contents = sio.loadmat('exported_ann_structure.mat',struct_as_record=False)
    #print(mat_contents)
    structure = mat_contents['exported_ann_structure']
    #print(my_ann_structure[0,0].input_xmin)


    while True:
        cc=str(ser.readline())
        #print(cc[2:][:-5])
        words = cc.split() # parse the string for the ratios
        if len(words) == 7:
            r_rat = float(words[1])
            g_rat = float(words[3])
            b_rat = float(words[5])

            ### this is where we get the arduino data through serial: 
            RGBratios = np.array([[r_rat, g_rat, b_rat]]) # use [[ ]] to make a 2d array, for compatible matrix multiplication.
            ### 

            # now, using the neural network and an RGB ratios input triplet, predict the rotation matrix: 
            predicted_rotation_matrix = my_ann_evaluation(structure, np.transpose(RGBratios))
            print(predicted_rotation_matrix)



if __name__ == "__main__": 
    main()
