from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import style

import numpy as np
import numpy.core.multiarray
import warnings
import tkinter as tk
from tkinter import ttk
import cv2
from skimage import io

import sr_main_keras as s_code

style.use('ggplot')
warnings.simplefilter('ignore')

from tkinter import filedialog


global_filename = ""
# ===========================================================================================
def get_input(inp):
    print(inp)


# function to browse files
def browsefunc():
    global global_filename
    filename = filedialog.askopenfilename()
    global_filename = filename
    pathlabel.config(text=filename)


# given the path to image, returns its name
def get_img_name(path):
    path_split = path.split("/")
    return path_split[-1]


# save the genrated image
def save_file(image, img_path, scale):
    img_name = get_img_name(img_path)
    save_img_name = img_name[:-4] + "_SR_x{0}".format(scale) + img_name[-4:]

    save_folder =  filedialog.askdirectory()
    save_file = save_folder + "/" + save_img_name

    io.imsave(save_file, image)


# function to Show low resolution image on a new pop up window
def show_lr(path):
    popup_lr = tk.Tk()
    popup_lr.wm_title("Low Resolution Image")

    label = ttk.Label(popup_lr, justify=tk.LEFT, text="""Original Low Resolution Image""", font=("Verdana", 14, "bold"))
    label.pack(side="top", fill="x", pady=30, padx=30)

    img = io.imread(path)
    if img is None:
        print(path)
        print(type(path))
        print("IMG IS NONE")
        
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    fig, ax = plt.subplots()
    im = ax.imshow(img, origin='upper')
    plt.grid("off")

    canvas = FigureCanvasTkAgg(fig, popup_lr)
    canvas.show()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2TkAgg(canvas, popup_lr)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    label = ttk.Label(popup_lr, justify=tk.CENTER, text="")
    label.pack(side="top", pady=2, padx=30)

    B1 = ttk.Button(popup_lr, text="SELECT FOLDER TO SAVE THIS IMAGE", command=lambda: save_file(img, path, scale=1))
    B1.pack(side="top")

    label = ttk.Label(popup_lr, justify=tk.CENTER, text="")
    label.pack(side="top", pady=2, padx=30)

    B2 = ttk.Button(popup_lr, text="CLOSE THIS WINDOW", command=popup_lr.destroy)
    B2.pack(side="top")

    popup_lr.mainloop()


# function to Show super resolved image on a new pop up window
def show_sr(path, scale=2):
    popup_sr = tk.Tk()
    if scale == 2:
        popup_sr.wm_title("Super Resolved Image x2")

        label = ttk.Label(popup_sr, justify=tk.CENTER, text="""Super Resoved Image x2""", font=("Verdana", 14, "bold"))
        label.pack(side="top", fill="x", pady=10, padx=30)
    elif scale == 4:
        popup_sr.wm_title("Super Resolved Image x4")

        label = ttk.Label(popup_sr, justify=tk.LEFT, text="""Super Resoved Image x4""", font=("Verdana", 14, "bold"))
        label.pack(side="top", fill="x", pady=12, padx=30)

    class_var = s_code.SuperResolution()
    class_var.create()

    if scale == 2:
        img = class_var.predict(path_=[path])
    if scale == 4:
        img = class_var.predict(path_=[path])
        img = class_var.predict(image=img)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    fig, ax = plt.subplots()
    im = ax.imshow(img, origin='upper')
    plt.grid("off")

    canvas = FigureCanvasTkAgg(fig, popup_sr)
    canvas.show()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2TkAgg(canvas, popup_sr)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    label = ttk.Label(popup_sr, justify=tk.CENTER, text="")
    label.pack(side="top", pady=2, padx=30)

    B1 = ttk.Button(popup_sr, text="SELECT FOLDER TO SAVE THIS IMAGE", command=lambda: save_file(img, path, scale=scale))
    B1.pack(side="top")

    label = ttk.Label(popup_sr, justify=tk.CENTER, text="")
    label.pack(side="top", pady=2, padx=30)
    
    B2 = ttk.Button(popup_sr, text="CLOSE THIS WINDOW", command = popup_sr.destroy)
    B2.pack(side= "top")

    popup_sr.mainloop()

# ============================================================================================

root = tk.Tk()
tk.Tk.wm_title(root, "Super Resolution GUI")
label = ttk.Label(root, text="Welcome to the Super Resolution GUI", font=("Verdana", 22, "bold"))
label.pack(side="top", pady=30, padx=50)

desc = '''This GUI allows people to use the Super Resolution Model with ease.
All you need to do is to drag and drop, and the rest would be managed by the GUI.'''
label = ttk.Label(root, justify=tk.CENTER, text=desc, font=("Verdana", 11))
label.pack(side="top", pady=30, padx=30)

label = ttk.Label(root, justify=tk.CENTER,
                  text="Click the browse button below to select the image file", font=("Verdana", 11))
label.pack(side="top", pady=5, padx=30)


button1 = ttk.Button(root, text="BROWSE", command=lambda: browsefunc())
button1.pack()

label = ttk.Label(root, justify=tk.CENTER, text="Path of the selected image file", font=("Verdana", 11))
label.pack(side="top", pady=3, padx=30)

pathlabel = ttk.Label(root, font=("Verdana", 11, "bold"))
pathlabel.pack(side="top", pady=3, padx=30)

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=1, padx=30)

button1 = ttk.Button(root, text="SHOW ORIGINAL IMAGE", command=lambda: show_lr(global_filename))
button1.pack()

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=2, padx=30)

button2 = ttk.Button(root, text="SUPER RESOLOVE X2", command=lambda: show_sr(global_filename, scale=2))
button2.pack()

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=2, padx=30)

button3 = ttk.Button(root, text="SUPER RESOLOVE X4", command=lambda: show_sr(global_filename, scale=4))
button3.pack()

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=2, padx=30)


button3 = ttk.Button(root, text="QUIT", command=lambda: show_sr(exit(0)))
button3.pack()

label = ttk.Label(root, justify=tk.CENTER, text="")
label.pack(side="top", pady=5, padx=30)

if __name__ == "__main__":
    root.mainloop()
