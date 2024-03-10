import os
import pickle
import tkinter as tk
from tkinter import Entry, Label, Button, Text, Scrollbar, messagebox,ttk
import cv2
from PIL import Image, ImageTk
from mediapipe.python.solutions import hands
import numpy as np
import mediapipe as mp
from src.train import train_hand_gesture_model
from src.pridiction import perform_hand_gesture_recognition
import threading

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class CollectDataTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.progress_var = tk.DoubleVar()
        self.setup_ui()
        self.hands_module = hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.class_labels = []
        # self.progress_var = tk.DoubleVar()
        # self.progress_var.set(0.0)
        
        

    def setup_ui(self):
        self.grid_columnconfigure((0, 1, 2), weight=1)

        self.camera_label = tk.Label(self, bg="black")
        self.camera_label.grid(row=0, column=0, columnspan=2, rowspan=3, padx=20, pady=20, sticky="nsew")

        data_entry_frame = tk.Frame(self, width=400)
        data_entry_frame.grid(row=0, column=2, rowspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        model_name_label = tk.Label(data_entry_frame, text="Model Name")
        model_name_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")

        self.model_name_entry = Entry(data_entry_frame)
        self.model_name_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        sign_label = Label(data_entry_frame, text="Sign Name")
        sign_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

        self.sign_entry = Entry(data_entry_frame)
        self.sign_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        submit_button = Button(data_entry_frame, text="Submit", command=self.collect_data)
        submit_button.grid(row=2, column=0, columnspan=2, pady=20, sticky="nsew")

        Train_button = Button(data_entry_frame, text="Train Model", command=self.train_data)
        Train_button.grid(row=4, column=0, columnspan=2, pady=(10, 10), sticky="nsew")

        self.progress_bar = ttk.Progressbar(data_entry_frame, orient="horizontal", length=200, mode="determinate", variable=self.progress_var)
        self.progress_bar.grid(row=5, column=0, columnspan=2, pady=(10, 10), sticky="nsew")

        # Pridict_button = Button(data_entry_frame, text="Predict Model", command=self.predict_data)
        # Pridict_button.grid(row=5, column=0, columnspan=2, pady=(10, 10), sticky="nsew"))

        Pridict_button = Button(data_entry_frame, text="Pridict Model", command=self.pridict_data)
        Pridict_button.grid(row=6, column=0, columnspan=2, pady=(10,10), sticky="nsew")

        self.print_output_text = Text(data_entry_frame, wrap="word", height=11, state="disabled", width=15)
        self.print_output_text.grid(row=3, column=0, columnspan=2, pady=(10, 10), sticky="nsew")

        self.capture = cv2.VideoCapture(0)
        self.update_camera()

    def collect_data(self):
        model_name = self.model_name_entry.get()
        sign_name = self.sign_entry.get()

        # Replace spaces with underscores in the sign name
        sign_name = sign_name.replace(" ", "_")

        if not model_name or not sign_name:
            messagebox.showwarning("Input Error", "Please enter Model Name and Sign Name.")
            return

        model_path = os.path.join("data", model_name)
        sign_path = os.path.join(model_path, sign_name)

        os.makedirs(model_path, exist_ok=True)
        os.makedirs(sign_path, exist_ok=True)
        self.class_labels.append(sign_name)

        count = 0
        while count < 100:
            ret, frame = self.capture.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands_module.process(rgb_frame)

                if results.multi_hand_landmarks:
                    image_filename = f"hand_{count}.png"
                    image_path = os.path.join(sign_path, image_filename)
                    cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    count += 1

                    self.print_output_text.config(state="normal")
                    self.print_output_text.insert(tk.END, f"Image {count} collected\n")
                    self.print_output_text.config(state="disabled")
                    self.print_output_text.see(tk.END)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)
                self.camera_label.configure(image=frame)
                self.camera_label.image = frame

                self.update_idletasks()
                self.after(10)

        messagebox.showinfo("Data Collection", "Data collection completed.")

        # with open(os.path.join("model", model_name+'.txt'), 'w') as f:
        #     for idx, label in enumerate(self.class_labels):
        #         f.write('{} {}\n'.format(label, idx))

    # def train_data(self):
    #     data = self.model_name_entry.get()
    #     if not data:
    #         messagebox.showwarning("Input Error", "Please enter Model Name.")
    #         return
    #     train_hand_gesture_model(data_path="data/"+data, model_folder="model", model_name=data+".p")
                
    def train_data(self):
        model_name = self.model_name_entry.get()
        if not model_name:
            messagebox.showwarning("Input Error", "Please enter Model Name.")
            return

        # Disable the Train button and enable the Progressbar
        self.disable_train_button()
        self.progress_bar.start()

        # Run the training function in a separate thread
        thread = threading.Thread(target=self.train_hand_gesture_model_thread, args=(model_name,))
        thread.start()

    def disable_train_button(self):
        for widget in self.winfo_children():
            if isinstance(widget, Button) and widget["text"] == "Train Model":
                widget.config(state=tk.DISABLED)

    def enable_train_button(self):
        for widget in self.winfo_children():
            if isinstance(widget, Button) and widget["text"] == "Train Model":
                widget.config(state=tk.NORMAL)

    def train_hand_gesture_model_thread(self, model_name):
        train_hand_gesture_model(data_path="data/" + model_name, model_folder="model", model_name=model_name + ".p")

        # Enable the Train button and stop the Progressbar after completing the function
        self.enable_train_button()
        self.progress_bar.stop()

        # Show a popup message with the accuracy
        messagebox.showinfo("Model Training Completed", f"Model training completed. Model saved as {model_name}.p.")


    def pridict_data(self):
        model_name = self.model_name_entry.get()
        if not model_name:
            messagebox.showwarning("Input Error", "Please enter Model Name.")
            return
        perform_hand_gesture_recognition("model/"+model_name+'.p', "model/"+model_name+'.txt')
        
    
    def update_camera(self):
        ret, frame = self.capture.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            self.camera_label.configure(image=frame)
            self.camera_label.image = frame
            self.after(10, self.update_camera)


if __name__ == "__main__":
    root = tk.Tk()
    collect_data_tab = CollectDataTab(root)
    collect_data_tab.pack(expand=True, fill="both")
    root.mainloop()
