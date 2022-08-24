"""
Main file to run the GUI

"""
import gui
import tkinter as tk

root = tk.Tk()
root.title('Simulation')
gui = gui.GUI(master=root)

root.mainloop()
