# ChromoSense
Code for the ChromoSense project: a multimodal deformation and temperature sensor. 
We include several directories with code that cannot be executed on Code Ocean. 
---------------------------------------------------------------------------------

digital_twin: contains the python scripts for generating the origami interface digital twin. 
run all scripts immediately within this directory with with python3 XXX.py, where XXX is the file name. 

-main.py runs a visualization of the workspace

-NN_prediction.py interfaces with the physical hardware (origami interface with ChromoSense sensor) and Arduino code included in this repository to generate a prediction for end effector position based on sensor input, and updates a graphical representation shown with Tkinter. The neural network structure it uses is exported_ann_structure.mat, generated in matlab from training data. 

tcs34725_rob_edit: contains the Arduino files (to be loaded onto Arduino Uno) required to read data from the utilized spectral sensor. See https://www.adafruit.com/product/1334 for more information on the sensor and example code. 


