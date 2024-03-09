import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import sys
from src.collect_data import CollectDataTab
# from crete_datasets import Embbedings

sys.path.append('Forest-ttk-theme-master')

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SignVision: Intelligent Hand Gesture Analysis")
        self.geometry("1100x580")
        self.resizable(width=False, height=False)
        ico = Image.open('images\logo.jpg')
        photo = ImageTk.PhotoImage(ico)
        self.wm_iconphoto(False, photo)

        style = ttk.Style()

        # Import the tcl file
        style.tk.call("source", "Forest-ttk-theme-master\\forest-dark.tcl")
        style.theme_use('forest-dark')

        self.tabview = ttk.Notebook(self,style="TNotebook")
        self.tabview.pack(pady=30,padx=30,fill="both",expand=True)

        CollectData = CollectDataTab(self.tabview)
        # self.collect_Data_tab(CollectData)
        self.tabview.add(CollectData, text="Collect Data")



if __name__ == "__main__":
    app = App()
    app.mainloop()