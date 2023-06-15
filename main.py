import customtkinter
import numpy as np
import win32gui
from PIL import ImageGrab
from customtkinter import CTkLabel, CTkCanvas, CTkButton, CTkFrame
from keras import models


def predict_digit(img):
    # resize image to 28x28 pixels
    img = img.resize((28, 28))
    # convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    # reshaping to support our model input and normalizing
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    # predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


# load the trained model from pickle file
model = models.load_model("digit_recognition_model")


class MainWindow:

    def __init__(self, window):
        self.x = self.y = 0

        window.title("BAD AI")
        frame = CTkFrame(window, border_width=2, border_color="cyan")
        frame.grid(row=0, column=0, sticky="NESW")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.pack(fill='both', expand=1)
        frame.place(in_=window, anchor="center", relx=.5, rely=.5)

        self.heading = CTkLabel(frame, text="Digit & Alphabet Recognition", font=("Gabriola", 40))
        self.canvas = CTkCanvas(frame, bg="white", height=300, width=300, cursor="cross")
        self.note = CTkLabel(frame, text="Draw Digit/Alphabet to recognize!", font=("Corbel Light", 30))
        self.rec_btn = CTkButton(frame, text="Recognize", font=("Corbel Light", 20), command=self.classify_handwriting)
        self.clear_btn = CTkButton(frame, text="Clear", font=("Corbel Light", 20), command=self.clear_all)
        self.compute = CTkLabel(frame, text="", font=("Corbel Light", 40))

        self.heading.grid(row=0, column=0, rowspan=2, columnspan=5, padx=20, pady=30)
        self.note.grid(row=2, column=0, rowspan=1, columnspan=2, padx=20)
        self.canvas.grid(row=3, column=0, rowspan=3, columnspan=2, padx=20, pady=20)
        self.rec_btn.grid(row=2, column=3, padx=20)
        self.clear_btn.grid(row=2, column=4, padx=20)
        self.compute.grid(row=3, column=3, columnspan=2, padx=20)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        canvas_handle = self.canvas.winfo_id()
        img_coords = win32gui.GetWindowRect(canvas_handle)
        digit, acc = predict_digit(ImageGrab.grab(img_coords))
        self.compute.configure(text="The digit is " + str(digit) + " with " + str(int(acc * 100)) + "% accuracy")

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 20
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


# Run the application
customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")
tk_win = customtkinter.CTk()
win = MainWindow(tk_win)

width = 1366
height = 768
x = (tk_win.winfo_screenwidth() / 2) - (width / 2)
y = (tk_win.winfo_screenheight() / 2) - (height / 2)
tk_win.geometry('%dx%d+%d+%d' % (width, height, x, y))
tk_win.mainloop()
