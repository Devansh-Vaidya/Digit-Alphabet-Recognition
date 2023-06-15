import customtkinter
import win32gui
from PIL import ImageGrab
from customtkinter import CTkLabel, CTkCanvas, CTkButton, CTkFrame


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
        self.canvas = CTkCanvas(frame, bg="white", height=400, width=400, cursor="cross")
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
        img = ImageGrab.grab(img_coords).resize((28, 28))
        self.compute.configure(text="Computing...")
        print(img.size)

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 10
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black')


# Run the application
customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")
tk_win = customtkinter.CTk()
win = MainWindow(tk_win)
tk_win.title('BAD AI')
screen_width = tk_win.winfo_screenwidth()  # Width of the screen
screen_height = tk_win.winfo_screenheight()  # Height of the screen

width = 1366
height = 768
x = (screen_width / 2) - (width / 2)
y = (screen_height / 2) - (height / 2)
tk_win.geometry('%dx%d+%d+%d' % (width, height, x, y))
tk_win.mainloop()
