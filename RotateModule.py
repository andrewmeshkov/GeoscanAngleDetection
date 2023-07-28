import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImagePath
import math
import numpy as np
from numpy import array as npa
import os

from clusters3 import CVAdapter
import cv2
import math as m


class RotateImageInterface(tk.Toplevel):
    def __init__(self, images, x, y, title, master: tk.Tk = None):
        super().__init__(master)

        self.geometry(f'{x}x{y}')

        self.rotation = CVAdapter(api_key="Ize9gbIm7pj9vTBIrPz7", model="vineyard-2lsq6", version=1, confidence=55)
        self.pil_image = None
        self.my_title = title
        self.theme_color = "white"
        self.able_line = False

        self.title(self.my_title)

        self.first_point = None
        self.index = 0

        self.images = images
        self.rotate_images()
        self.first_images = []

        # cv2.imshow('', self.images[0])
        self.create_widget()
        self.create_menu()
        self.reset_transform()
        for img in images:
            ind = len([name for name in os.listdir("C:\\Geoscan\\subfield\\")])
            print(ind)
            cv2.imwrite(f'C:\\Geoscan\\subfield\\{ind}.png', img)


    def rotate_images(self):

        for i in range(len(self.images)):
            preproc = self.images[i]
            cv2.imwrite(f'C:\\Geoscan\\term\\{i}.png', preproc)
            angle = self.rotation.angle(f"C:\\Geoscan\\term\\{i}.png")
            height, width = preproc.shape[:2]
            center = (width / 2, height / 2)
            rotate_matrix = cv2.getRotationMatrix2D(center=center,
                                                    angle=angle,
                                                    scale=1)
            rotated_image = cv2.warpAffine(src=preproc, M=rotate_matrix, dsize=(preproc.shape[:2]))
            self.images[i] = rotated_image


    def menu_open_clicked(self, event=None):
        filename = tk.filedialog.askopenfilename(filetypes=[("Image file", ".bmp .png .jpg .JPG .tif .TIF .tiff .TIFF"),
                                                            ("Bitmap", ".bmp"),
                                                            ("PNG", ".png"),
                                                            ("JPEG", ".jpg .JPG"),
                                                            ("Tiff", ".tif .TIF .tiff .TIFF")],
                                                 initialdir=os.getcwd())
        self.set_image(filename)

    def menu_quit_clicked(self):
        self.destroy()

    def create_menu(self):
        self.menu_bar = tk.Menu(self)

        self.file_menu = tk.Menu(self.menu_bar, tearoff=tk.OFF)
        self.menu_bar.add_cascade(label="Файл", menu=self.file_menu)
        self.file_menu.add_command(label="Сохранить", command=self.save_image)

        self.image_menu = tk.Menu(self.menu_bar, tearoff=tk.OFF)
        self.menu_bar.add_cascade(label="Изображения", menu=self.image_menu)

        for i in range(len(self.images)):
            self.image_menu.add_command(label="Image " + str(i + 1), command=lambda i=i: self.change_image(i),
                                        accelerator="Ctrl+O")
            self.image_menu.add_separator()

        self.menu_bar.bind_all("<Control-o>", self.menu_open_clicked)
        self.config(menu=self.menu_bar)

    def save_image(self):
        cv2.imwrite("C:\\Geoscan\\rotatedSubfields\\" + str(len(os.listdir("C:\\Geoscan\\rotatedSubfields"))) + ".png", self.images[self.index])

    def change_image(self, ind):
        print(ind)
        self.index = ind
        self.pil_image = Image.fromarray(self.images[ind])
        self.redraw_image()

    def create_widget(self):

        self.canvas = tk.Canvas(self, background="white")
        self.canvas.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        frame_statusbar = tk.Frame(self, bd=1, relief=tk.SUNKEN)
        # self.next = tk.Button(self, text="Построить ряд", command=self.make_row)
        # self.next.pack(side=tk.TOP)
        self.label_image_info = tk.Label(frame_statusbar, text="image info", anchor=tk.E, padx=5)
        self.label_image_pixel = tk.Label(frame_statusbar, text="(x, y)", anchor=tk.W, padx=5)
        self.label_image_info.pack(side=tk.RIGHT)
        self.label_image_pixel.pack(side=tk.LEFT)
        frame_statusbar.pack(side=tk.TOP, fill=tk.X)

        self.canvas.bind("<Button-1>", self.mouse_down_left)
        # self.master.bind("<ButtonRelease-1>", self.mouse_up)
        self.canvas.bind("<Double-Button-1>", self.make_row)
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

        self.title(self.my_title + " - " + os.path.basename(filename))
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

    def put_points(self):
        if self.able_line:
            pass

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
        self.redraw_image()

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

    def make_row(self, event):
        self.able_line = True
        if self.first_point == None:
            self.first_point = [event.x, event.y]
        else:
            self.second_point = [event.x, event.y]
            height, width = self.images[self.index].shape[:2]
            center = (width / 2, height / 2)
            if self.second_point[0] - self.first_point[0] == 0:
                angles = 90
            else:
                angles = m.degrees(m.atan((self.second_point[1] - self.first_point[1]) / (
                        self.second_point[0] - self.first_point[0])))
            print(angles)
            '''rotate_matrix = cv2.getRotationMatrix2D(center=center,
                                                    angle=angles,
                                                    scale=1)
            rotated_image = cv2.warpAffine(src=self.images[self.index], M=rotate_matrix, dsize=(width, height))
            cv2.imwrite('RotatedSubfield.png', rotated_image)
            self.pil_image = Image.fromarray(rotated_image)
            self.images[self.index] = rotated_image
            self.redraw_image()
            self.first_point = None
            self.able_line = False'''
            self.first_point = None

    def redraw_image(self):
        if self.pil_image is None:
            return
        self.draw_image(self.pil_image)
