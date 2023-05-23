import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from src.ImageColorization import colorize

def resize_all(app):
    w = app.window.winfo_width()
    h = app.window.winfo_height()
    if (app.image1 != None):
        app.image1 = app.image1.resize(((int)(w * 0.45), (int)(h * 0.65)))  # Adjust the size of the image as needed
        app.image1_tk = ImageTk.PhotoImage(app.image1)
        app.image_label1.configure(image=app.image1_tk)
        app.image_label1.image = app.image1_tk
    if app.image2 != None:
        app.image2 = app.image2.resize(((int)(w * 0.45), (int)(h * 0.65)))
        app.image2_tk2 = ImageTk.PhotoImage(app.image2)
        app.image_label2.configure(image=app.image2_tk2)
        app.image_label2.image = app.image2_tk2

# Create the main window
class App:
    def open_image(self, image_label):
        filepath = filedialog.askopenfilename(title="Open Image",
                                          filetypes=(("Image files", "*.jpg;*.jpeg;*.png"),
                                                     ("All files", "*.*")))
        if filepath:
            w = self.window.winfo_width()
            h = self.window.winfo_height()
            self.image1 = Image.open(filepath)
            self.image1 = self.image1.resize(((int)(w * 0.45), (int)(h * 0.65)))  # Adjust the size of the image as needed
            self.image1_tk = ImageTk.PhotoImage(self.image1)
            self.image_label1 = image_label
            self.image_label1.configure(image=self.image1_tk)
            self.image_label1.image = self.image1_tk

    def color_image(self):
        w = self.window.winfo_width()
        h = self.window.winfo_height()
        self.image2 = colorize(self.image1)
        self.image2 = self.image2.resize(((int)(w * 0.45), (int)(h * 0.65)))
        self.image2_tk = ImageTk.PhotoImage(self.image2)
        self.image_label2.configure(image=self.image2_tk)
        self.image_label2.image = self.image2_tk


    def __init__(self) -> None:  
        #create the Null varaibles
        self.image1 = None
        self.image1_tk = None
        self.image2 = None
        self.image2_tk = None

        # Create the main window
        self.window = tk.Tk()
        self.window.title("Image Viewer")
        self.window.geometry("640x360")
        self.window.bind("<Configure>", lambda e: resize_all(self))
        
        # Create a button to open the image
        self.open_button = tk.Button(self.window, text="Open Image", command=lambda: self.open_image(self.image_label1), padx=10, pady=10)
        self.open_button.place(relx=0.78, rely=0.95, anchor="se")

        # Create a second button to colorize image
        self.colorize_button = tk.Button(self.window, text="Colorize Image", command=lambda: self.color_image(), padx=10, pady=10)
        self.colorize_button.place(relx=0.97, rely=0.95, anchor="se")

        # Create a label to display the image
        self.image_label1 = tk.Label(self.window)
        self.image_label1.pack(side="left")
        
        # Create a second label to display the colorized image
        self.image_label2 = tk.Label(self.window)
        self.image_label2.pack(side="left")

        # Start the Tkinter event loop
        self.window.mainloop()


if (__name__ == "__main__"):
    app = App()