import customtkinter
import cv2
import numpy as np
from PIL import ImageDraw, Image
from keras import models

by_merge_map = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B',
                12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M',
                23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X',
                34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q',
                45: 'r', 46: 't'}


def predict_digit(img):
    # convert to grayscale, and resize to 28x28 pixel
    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    img_array = cv2.resize(img_array, (28, 28))

    # reshaping to support our model input and normalizing
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    # predicting the class
    res = model.predict([img_array])[0]
    return np.argmax(res), max(res)


# load the trained model from pickle file
model = models.load_model("emnist_merge_recognition_model")


class MainWindow:

    def __init__(self, window):
        self.x = self.y = 0

        window.title("BAD AI")
        frame = customtkinter.CTkFrame(window, border_width=2, border_color="cyan")
        frame.grid(row=0, column=0, sticky="NESW")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.pack(fill='both', expand=1)
        frame.place(in_=window, anchor="center", relx=.5, rely=.5)

        self.heading = customtkinter.CTkLabel(frame, text="Digit & Alphabet Recognition", font=("Gabriola", 40))
        self.canvas = customtkinter.CTkCanvas(frame, bg="white", height=400, width=400, cursor="cross")
        self.note = customtkinter.CTkLabel(frame, text="Draw Digit/Alphabet to recognize!", font=("Corbel Light", 30))
        self.rec_btn = customtkinter.CTkButton(frame, text="Recognize", font=("Corbel Light", 20), command=self.classify_handwriting)
        self.clear_btn = customtkinter.CTkButton(frame, text="Clear", font=("Corbel Light", 20), command=self.clear_all)
        self.predict = customtkinter.CTkLabel(frame, text="", font=("Corbel Light", 45))
        self.accurate = customtkinter.CTkLabel(frame, text="", font=("Corbel Light", 45))

        self.heading.grid(row=0, column=0, rowspan=2, columnspan=5, padx=20, pady=30)
        self.note.grid(row=2, column=0, rowspan=1, columnspan=2, padx=20)
        self.canvas.grid(row=3, column=0, rowspan=3, columnspan=2, padx=20, pady=20)
        self.rec_btn.grid(row=2, column=3, padx=20)
        self.clear_btn.grid(row=2, column=4, padx=20)
        self.predict.grid(row=3, column=3, columnspan=2, padx=20)
        self.accurate.grid(row=4, column=3, columnspan=2, padx=20)

        self.canvas.bind("<B1-Motion>", self.draw_lines)
        # Get Image from canvas
        self.img = Image.new('RGB', (400, 400), (0, 0, 0))
        self.img_draw = ImageDraw.Draw(self.img)

    def clear_all(self):
        self.canvas.delete("all")
        self.img = Image.new('RGB', (400, 400), (0, 0, 0))
        self.img_draw = ImageDraw.Draw(self.img)

    def classify_handwriting(self):
        prediction, acc = predict_digit(self.img)
        self.predict.configure(text="Predicted Value: " + str(by_merge_map[prediction]) + "!")
        self.accurate.configure(text="Accuracy: " + str(int(acc * 100)) + "%")

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 15
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')
        self.img_draw.ellipse((self.x - r, self.y - r, self.x + r, self.y + r), fill='white')


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
