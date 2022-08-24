"""
GUI to simulate an origami robot. 

This GUI file interacts with the main.py and module.py files.

"""

import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import numpy as np
import module
from tkinter import filedialog
import pandas as pd
import time
import serial
import scipy.io as sio
import cv2.aruco as aruco
import os

import matplotlib.animation as animation


class GUI(tk.Frame):
    """GUI Class to create and interact with the GUI

    Args:
        tk (tk.Frame): tkinter library framework to create and compose GUI.
    """

    def __init__(self, master=None):
        """Initialisation of the GUI

        Args:
            master (tk.TK(), optional): Root of the GUI, allows to declare the GUI on a single thread. Defaults to None.
        """
        super().__init__(master)
        self.master = master
        self.grid()
        self.create_widgets()

        self.value_res = 0

    def create_widgets(self):
        """Create the GUI widgets
        """

        # Workspace widget
        self.label_show_workspace = tk.Label(self.master, text='Plot workspace')
        self.label_show_workspace.grid(row=0, column=0, padx=30)
        self.var_plot_ws = tk.IntVar()

        # Workspace activation through a checkbutton
        self.plot_workspace_button = tk.Checkbutton(self.master, command=self.add_workspace, selectcolor='black',
                                                    variable=self.var_plot_ws)
        self.plot_workspace_button.grid(row=0, column=1, padx=30)
        self.ws_plotted = False

        self.label_show_workspace = tk.Label(self.master, text='Resolution workspace')
        self.label_show_workspace.grid(row=1, column=0, padx=30)
        self.resolution_workspace = tk.Scale(self.master, from_=1, to=100, command=self.adapt_resolution,
                                             orient=tk.HORIZONTAL)
        self.resolution_workspace.grid(row=2, column=0,columnspan=2, sticky=tk.NSEW, padx=30)

        # Quit button
        self.quit = tk.Button(self.master, text='QUIT', command=self.quit_program)
        self.quit.grid(row=13, column=0)

        # Module generation
        self.module = module.Module(length_base_arms=6, length_base_between_arms=10, length_lower_arm=5,
                                    length_upper_arm=10, length_surface_arms=6, length_surface_between_arms=10,
                                    thickness=0, joint_radius=0.1, translate_x=0, translate_y=0)

        # Initial figure generation
        self.figure_canvas()
        self.module.plot_module(self.ax)
        self.fig_tk.draw

        # Possibility to connect module through Serial connection. Change port according to your OS and the port you are connecting your origami to.
        self.ser = None
        try:
            self.ser = serial.Serial('/dev/ttyACM0', timeout=None, baudrate=57600)
        except serial.SerialException:
            pass

        # Initialize rotation matrix to compute inverse kinematics.
        self.predicted_rotation_matrix = None

        # Update according to serial connection.
        self.update()

    def adapt_resolution(self, value):
        self.value_res = int(value)

    def quit_program(self):
        """Quit nicely the program closing the serial connection if needed.
        """
        if self.ser is not None:
            self.ser.close()
        self.master.destroy()

    def add_workspace(self):
        """Add workspace plot to simulation. Verify first if workspace is not plotted.
        """
        self.ws_plotted = not (self.ws_plotted);

    def update(self, workspace=False):
        """Update the simulation by computing a new rotation matrix according to sensors inputs.
        """
        if self.predicted_rotation_matrix is None:
            self.predicted_rotation_matrix = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        if self.ser is not None:
            # first, load our neural network structure that we exported from matlab. 
            mat_contents = sio.loadmat('exported_ann_structure.mat', struct_as_record=False)

            structure = mat_contents['exported_ann_structure']

            cc = str(self.ser.readline())
            self.ser.flushInput()
            words = cc.split()  # parse the string for the ratios
            if len(words) == 7:
                r_rat = float(words[1])
                g_rat = float(words[3])
                b_rat = float(words[5])

                ### this is where we get the arduino data through serial: 
                RGBratios = np.array(
                    [[r_rat, g_rat, b_rat]])  # use [[ ]] to make a 2d array, for compatible matrix multiplication.
                ### 

                # now, using the neural network and an RGB ratios input triplet, predict the rotation matrix: 
                self.predicted_rotation_matrix = \
                self.my_ann_evaluation(structure, np.transpose(RGBratios)).transpose().tolist()[0]
                self.predicted_rotation_matrix = list(self.predicted_rotation_matrix)

            # Inverse kinematics computation according to newly computed rotation matrix of the surface.
            self.module.inverse_kinematics(self.predicted_rotation_matrix)

        # Reset axis to draw updated module
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_xlim(15, -15)
        self.ax.set_ylim(-15, 15)
        self.ax.set_zlim(0, 20)
        if self.var_plot_ws.get() or self.ws_plotted:
            self.module.plot_workspace(self.ax, self.value_res)
            self.ws_plotted = True
        else:
            self.ws_plotted = False
        self.module.plot_module(self.ax)
        self.fig_tk.draw()

        # Call the update function every 1 ms
        self.after(100, self.update)

    def my_ann_evaluation(self, my_ann_structure, input):
        """Evaluation of ann structure to retrieve rotation matrix.

        Args:
            my_ann_structure (matlab matrix exported): Neural network
            input (np.array): 1x1x3 array containing R, G, B ratios.

        Returns:
            np.array: Rotation matrix with dimension 1x1x9
        """

        # extract fields as variables
        ymax = my_ann_structure[0, 0].input_ymax;
        ymin = my_ann_structure[0, 0].input_ymin;
        xmax = my_ann_structure[0, 0].input_xmax;
        xmin = my_ann_structure[0, 0].input_xmin;

        input_preprocessed = (ymax - ymin) * (input - xmin) / (xmax - xmin) + ymin;

        # Pass it through the ANN matrix multiplication
        y1 = np.tanh(np.matmul(my_ann_structure[0, 0].IW, input_preprocessed) + my_ann_structure[0, 0].b1);
        y2 = np.matmul(my_ann_structure[0, 0].LW, y1) + my_ann_structure[0, 0].b2;

        ymax = my_ann_structure[0, 0].output_ymax;
        ymin = my_ann_structure[0, 0].output_ymin;
        xmax = my_ann_structure[0, 0].output_xmax;
        xmin = my_ann_structure[0, 0].output_xmin;

        res = (y2 - ymin) * (xmax - xmin) / (ymax - ymin) + xmin;

        # return the result of the neural network:
        return res

    def figure_canvas(self):
        """Figure Canvas to embed on tkinter frame.
        """
        self.fig = plt.Figure(figsize=(20, 20), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.axis('off')
        self.fig_tk = FigureCanvasTkAgg(self.fig, self.master)
        self.fig_tk.get_tk_widget().grid(row=0, column=2, rowspan=13)
