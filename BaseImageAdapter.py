import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImagePath
import math
import numpy as np
from numpy import array as npa
import os
from rowsfinderwindow import RowViewer
import cv2
import math as m
from random import randrange


class BaseImageAdapter(tk.Frame):
    def __init__(self, x, y, title, color, master: tk.Tk = None):
        super().__init__(master)

        self.master.geometry(f'{x}x{y}')

        self.pil_image = None
        self.my_title = title
        self.theme_color = color

        self.master.title(self.my_title)

        self.create_menu()
        self.create_widget()
        self.reset_transform()


    def menu_open_clicked(self, event=None):
        filename = tk.filedialog.askopenfilename(filetypes=[("Image file", ".bmp .png .jpg .JPG .tif .TIF .tiff .TIFF"),
                                                            ("Bitmap", ".bmp"),
                                                            ("PNG", ".png"),
                                                            ("JPEG", ".jpg .JPG"),
                                                            ("Tiff", ".tif .TIF .tiff .TIFF")],
                                                 initialdir=os.getcwd())
        self.set_image(filename)

    def menu_quit_clicked(self):
        self.master.destroy()

    def create_menu(self):
        self.menu_bar = tk.Menu(self)

        self.file_menu = tk.Menu(self.menu_bar, tearoff=tk.OFF)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.file_menu.add_command(label="Open", command=self.menu_open_clicked, accelerator="Ctrl+O")
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.menu_quit_clicked)
        self.menu_bar.bind_all("<Control-o>", self.menu_open_clicked)
        self.master.config(menu=self.menu_bar)

    def create_widget(self):

        self.canvas = tk.Canvas(self.master, background="white")
        self.canvas.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        frame_statusbar = tk.Frame(self.master, bd=1, relief=tk.SUNKEN)
        self.label_image_info = tk.Label(frame_statusbar, text="image info", anchor=tk.E, padx=5)
        self.label_image_pixel = tk.Label(frame_statusbar, text="(x, y)", anchor=tk.W, padx=5)
        self.label_image_info.pack(side=tk.RIGHT)
        self.label_image_pixel.pack(side=tk.LEFT)
        frame_statusbar.pack(side=tk.TOP, fill=tk.X)

        self.canvas.bind("<Button-1>", self.mouse_down_left)
        # self.master.bind("<ButtonRelease-1>", self.mouse_up)
        self.canvas.bind("<B1-Motion>", self.mouse_move_left)
        self.canvas.bind("<Motion>", self.mouse_move)
        self.canvas.bind("<Button-2>", self.flash_zoom)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
    def set_image(self, filename):
        if not filename:
            return
        self.pil_image = Image.open(filename)
        self.zoom_fit(self.pil_image.width, self.pil_image.height)
        self.draw_image(self.pil_image)

        self.master.title(self.my_title + " - " + os.path.basename(filename))
        self.label_image_info[
            "text"] = f"{self.pil_image.format} : {self.pil_image.width} x {self.pil_image.height} {self.pil_image.mode}"
        os.chdir(os.path.dirname(filename))

    # def mouse_up(self, event = None):
    #	self.chosen_point = None

    def mouse_down_left(self, event):
        # print("mouse_down_left")
        # if self.closest_distance < 20**2:
        #	self.chosen_point = self.closest_point
        # else:
        #	self.chosen_point = None

        self.__old_event = event

    def mouse_move_left(self, event):
        if self.pil_image == None:
            return
        # if self.chosen_point is not None:
        self.translate(event.x - self.__old_event.x, event.y - self.__old_event.y)
        self.redraw_image()
        self.__old_event = event

    def mouse_move(self, event):
        if self.pil_image is None:
            return


        image_point = self.to_image_point(event.x, event.y)
        self.label_image_pixel["text"] = (f"({image_point[0]:.2f}, {image_point[1]:.2f})")
        self.label_image_pixel.configure(fg="black")
        if image_point[0] < 0 or image_point[1] < 0 or \
                image_point[0] > self.pil_image.width or image_point[1] > self.pil_image.height:
            self.label_image_pixel.configure(fg="red")

    # else:
    #	self.label_image_pixel["text"] = ("(--, --)")

    def flash_zoom(self, event):
        if self.pil_image is None:
            return
        self.zoom_fit(self.pil_image.width, self.pil_image.height)
        self.redraw_image()

    def mouse_wheel(self, event):
        ''' マウスホイールを回した '''
        if self.pil_image == None:
            return

        if event.state != 9:  # 9はShiftキー(Windowsの場合だけかも？)
            if (event.delta < 0):
                self.scale_at(1.25, event.x, event.y)
            else:
                # 上に回転の場合、縮小
                self.scale_at(0.8, event.x, event.y)
        else:
            if (event.delta < 0):
                # 下に回転の場合、反時計回り
                self.rotate_at(-5, event.x, event.y)
            else:
                # 上に回転の場合、時計回り
                self.rotate_at(5, event.x, event.y)
        self.redraw_image()  # 再描画

    def reset_transform(self):
        self.mat_affine = np.eye(3)

    def translate(self, offset_x, offset_y):
        mat = np.eye(3)
        mat[0, 2] = float(offset_x)
        mat[1, 2] = float(offset_y)
        self.mat_affine = mat @ self.mat_affine

    def scale(self, scale: float):
        self.mat_affine = np.diag([scale, scale, 1]) @ self.mat_affine

    def scale_at(self, scale: float, cx: float, cy: float):

        self.translate(-cx, -cy)
        self.scale(scale)
        self.translate(cx, cy)

    def zoom_fit(self, image_width, image_height):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if (image_width * image_height <= 0) or (canvas_width * canvas_height <= 0):
            return

        self.reset_transform()

        scale = 1.0
        offsetx = 0.0
        offsety = 0.0

        if (canvas_width * image_height) > (image_width * canvas_height):
            scale = canvas_height / image_height
            offsetx = (canvas_width - image_width * scale) / 2
        else:
            scale = canvas_width / image_width
            offsety = (canvas_height - image_height * scale) / 2

        self.scale(scale)
        self.translate(offsetx, offsety)

    def to_image_point(self, x, y):
        if self.pil_image is None:
            return []
        image_point = np.linalg.inv(self.mat_affine) @ np.array([x, y, 1.])
        return image_point

    def draw_image(self, pil_image):
        if pil_image is None:
            return

        self.pil_image = pil_image

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        mat_inv = np.linalg.inv(self.mat_affine)

        affine_inv = mat_inv[:-1, :].flatten()

        dst = self.pil_image.transform((canvas_width, canvas_height), Image.AFFINE, affine_inv,
                                       Image.NEAREST, fillcolor=(255, 255, 255, 0))

        self.image = ImageTk.PhotoImage(image=dst)
        # self.canvas.delete("all")
        item = self.canvas.create_image(0, 0, anchor='nw', image=self.image)

    def redraw_image(self):
        if self.pil_image is None:
            return
        self.draw_image(self.pil_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = BaseImageAdapter(x=1000,y=1000, color="black", title="oop rules", master=root)
    app.mainloop()
