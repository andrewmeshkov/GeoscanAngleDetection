import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImagePath
import math
import numpy as np
from numpy import array as npa
import os
import cv2
import math as m
from random import randrange
from RotateModule import RotateImageInterface


def get_distance_to_segment(point, segment):
	"""
	Функция вычисляет расстояние от точки до отрезка
	Args:
		point: Точка (x, y)
		segment: Отрезок, заданный двумя концами (x1, y1), (x2, y2)
	Returns: 
		Расстояние от точки до отрезка 
	""" 
	x, y = point
	x1, y1, x2, y2 = segment
	A = x - x1
	B = y - y1
	C = x2 - x1
	D = y2 - y1

	dot = A * C + B * D
	len_sq = C*C + D*D
	param = -1.0
	if len_sq != 0:
		param = dot / len_sq

	res = 0
	if param < 0:
		xx = x1
		yy = y1
		res = -1
	elif param > 1:
		xx = x2
		yy = y2
		res = 1
	else:
		xx = x1 + param * C
		yy = y1 + param * D

	dx = x - xx
	dy = y - yy
	return math.sqrt(dx*dx + dy*dy), res


class Application(tk.Frame):
	def __init__(self, master : tk.Tk = None):
		super().__init__(master)

		self.master.geometry("1280x720")

		self.pil_image = None
		self.my_title = "Image Viewer"
		self.theme_color = "white"

		self.master.title(self.my_title)

		self.create_menu()
		self.create_widget()
		self.reset_transform()
		self.images = []
		self.colors = [(255,0,0,100), (0,255,0,100), (0,0,255,100),
					   (255,0,255,100), (0,255,255,100), (153,153,255,100), (255,128255,0,100)]
		self.area = [[]]
		self.subfields = [[]]

		#self.chosen_point = None
		self.closest_point = None
		self.closest_distance = 10e300
		self.prev_closest_point = None
		self.prev_closest_distance = 10e300

		empty_im = Image.new("RGBA", (30, 30))
		ImageDraw.Draw(empty_im).ellipse([(0, 0), (20, 20)], fill=(255, 0, 0, 127))
		self.empty_im = ImageTk.PhotoImage(empty_im)

		circled_img = Image.new("RGBA", (30, 30))
		ImageDraw.Draw(circled_img).ellipse([(0, 0), (20, 20)], fill=(255, 0, 0, 200), outline=(100,0,0,255), width=3)
		self.circled_img = ImageTk.PhotoImage(circled_img)

		self.polygon_img = None

		self.i = 0

	def menu_open_clicked(self, event=None):
		filename = tk.filedialog.askopenfilename(filetypes=[("Image file", ".bmp .png .jpg .JPG .tif .TIF .tiff .TIFF"),
		                                                    ("Bitmap", ".bmp"),
		                                                    ("PNG", ".png"),
		                                                    ("JPEG", ".jpg .JPG"),
		                                                    ("Tiff", ".tif .TIF .tiff .TIFF")],
		                                         initialdir=os.getcwd())
		self.set_image(filename)
		self.area = [[]]
		self.subfields = [[]]
		self.images = []


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

		'''self.view_menu = tk.Menu(self.menu_bar, tearoff=tk.OFF)
		self.menu_bar.add_cascade(label="View", menu=self.view_menu)
		self.theme_menu = tk.Menu(self.view_menu, tearoff=tk.OFF)
		self.view_menu.add_cascade(label="Theme", menu=self.theme_menu)
		self.theme_menu.add_command(label="Dark",  command=lambda: self.set_theme("black"))
		self.theme_menu.add_command(label="Light", command=lambda: self.set_theme("white"))'''

		self.master.config(menu=self.menu_bar)

	def create_widget(self):

		self.canvas = tk.Canvas(self.master, background="white")
		self.canvas.pack(side = tk.TOP, expand=True, fill=tk.BOTH)

		self.btn_proceed = tk.Button(self.master, text="Исследовать", command=self.explore)
		self.btn_proceed.pack(side=tk.TOP)

		self.next = tk.Button(self.master, text="Далее", command=self.next)
		self.next.pack(side=tk.TOP)
		self.btn_proceed = tk.Button(self.master, text="Удалить", command=self.delete)
		self.btn_proceed.pack(side=tk.TOP)
		frame_statusbar = tk.Frame(self.master, bd=1, relief=tk.SUNKEN)
		self.label_image_info = tk.Label(frame_statusbar, text="image info", anchor=tk.E, padx=5)
		self.label_image_pixel = tk.Label(frame_statusbar, text="(x, y)", anchor=tk.W, padx=5)
		self.label_image_info.pack(side=tk.RIGHT)
		self.label_image_pixel.pack(side=tk.LEFT)
		frame_statusbar.pack(side=tk.TOP, fill=tk.X)





		self.canvas.bind("<Button-1>", self.mouse_down_left)
		#self.master.bind("<ButtonRelease-1>", self.mouse_up)
		self.canvas.bind("<B1-Motion>", self.mouse_move_left)
		self.canvas.bind("<Motion>", self.mouse_move)
		self.canvas.bind("<Double-Button-1>", self.put_point)
		self.canvas.bind("<Button-2>", self.flash_zoom)
		self.canvas.bind("<MouseWheel>", self.mouse_wheel)
		self.master.bind("<Delete>", self.delete_point)
		self.master.bind("<BackSpace>", self.delete_point)

	def set_image(self, filename):
		if not filename:
			return
		self.pil_image = Image.open(filename)
		self.zoom_fit(self.pil_image.width, self.pil_image.height)
		print(self.area)
		self.draw_image(self.pil_image)

		self.master.title(self.my_title + " - " + os.path.basename(filename))
		self.label_image_info["text"] = f"{self.pil_image.format} : {self.pil_image.width} x {self.pil_image.height} {self.pil_image.mode}"
		os.chdir(os.path.dirname(filename))

	#def mouse_up(self, event = None):
	#	self.chosen_point = None

	def mouse_down_left(self, event):
		#print("mouse_down_left")
		#if self.closest_distance < 20**2:
		#	self.chosen_point = self.closest_point
		#else:
		#	self.chosen_point = None

		self.__old_event = event

	def mouse_move_left(self, event):
		if self.pil_image == None:
			return
		#if self.chosen_point is not None:
		if self.closest_distance < 20 ** 2:
			mouse_point_coords = self.to_image_point(event.x, event.y)
			#self.chosen_point[1] = mouse_point_coords
			self.closest_point[1] = mouse_point_coords
		else:
			self.translate(event.x - self.__old_event.x, event.y - self.__old_event.y)
		self.redraw_image()
		self.__old_event = event

	def mouse_move(self, event):
		if self.pil_image is None:
			return

		self.refine_closest_point(event.x, event.y)

		if (self.closest_distance - 20**2) * (self.prev_closest_distance - 20**2) <= 0:
			self.redraw_image()
		elif (self.closest_distance - 20**2) <= 0 and (self.prev_closest_distance - 20**2) <= 0 and \
		     self.prev_closest_point != self.closest_point:
			self.redraw_image()

		image_point = self.to_image_point(event.x, event.y)
		self.label_image_pixel["text"] = (f"({image_point[0]:.2f}, {image_point[1]:.2f})")
		self.label_image_pixel.configure(fg="black")
		if image_point[0] < 0 or image_point[1] < 0 or \
		   image_point[0] > self.pil_image.width or image_point[1] > self.pil_image.height:
			self.label_image_pixel.configure(fg = "red")
		#else:
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

	def delete(self):
		self.area = [[]]
		self.subfields = [[]]
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
		                               Image.NEAREST, fillcolor=(255,255,255,0))

		self.image = ImageTk.PhotoImage(image=dst)
		#self.canvas.delete("all")
		item = self.canvas.create_image(0, 0, anchor='nw', image=self.image)
		field = []

		for i in range(len(self.area)):
			canvas_coords = []

			for point in self.area[i]:
				canvas_coords.append(tuple((self.mat_affine @ point[1])[0:2]))
				item = self.canvas.create_image(canvas_coords[-1][0] - 10, canvas_coords[-1][1] - 10, image=point[0],
												anchor='nw')
			field.append(canvas_coords)

		for i in range(len(self.area)):
			if len(field[i]) > 2:
				polygon_img = Image.new("RGBA", (self.image.width(), self.image.height()))
				ImageDraw.Draw(polygon_img).polygon(field[i], fill=self.colors[i],
													outline=self.colors[i])
				self.polygon_img = ImageTk.PhotoImage(polygon_img)
				item = self.canvas.create_image(0, 0, image=self.polygon_img, anchor='nw')

	def redraw_image(self):
		if self.pil_image is None:
			return
		self.draw_image(self.pil_image)

	def put_point(self, event):
		#print(f"put_point, event = {event}")
		if self.pil_image is None or self.closest_distance < 20**2:
			return

		self.subfields[-1].append([m.floor(self.to_image_point(event.x, event.y)[0]), m.floor(self.to_image_point(event.x, event.y)[1])])

		min_dist = 1e100
		ind = len(self.area[len(self.area)-1])
		insert = ind
		closest = 0
		point_coords = self.to_image_point(event.x, event.y)
		#print(f"point_coords = {point_coords}, event = {event.x, event.y}")
		added_point = [self.circled_img, point_coords]
		if len(self.area[len(self.area)-1]) > 2:
			for i, point in enumerate(self.area[len(self.area)-1]):
				point[0] = self.empty_im
				first = point[1][0:2]
				second = self.area[len(self.area)-1][(i+1)%len(self.area[len(self.area)-1])][1][0:2]

				curr_dist, curr_closest = get_distance_to_segment(point_coords[0:2], [*first, *second])

				if curr_dist < min_dist:
					min_dist = curr_dist
					closest = curr_closest
					ind = i+1
					insert = ind

			if closest != 0:
				if closest == -1:
					ind_x1 = (ind-2)%len(self.area[len(self.area)-1])
					ind_x2 = (ind-1)%len(self.area[len(self.area)-1])
					ind_x3 = (ind-0)%len(self.area[len(self.area)-1])
				elif closest == 1:
					ind_x1 = (ind - 1) % len(self.area[len(self.area)-1])
					ind_x2 = (ind + 0) % len(self.area[len(self.area)-1])
					ind_x3 = (ind + 1) % len(self.area[len(self.area)-1])

				x2x1 = (npa(self.area[len(self.area)-1][ind_x1][1]) - npa(self.area[len(self.area)-1][ind_x2][1]))[0:2]
				x2x3 = (npa(self.area[len(self.area)-1][ind_x3][1]) - npa(self.area[len(self.area)-1][ind_x2][1]))[0:2]
				x2xp = (npa(point_coords) - npa(self.area[len(self.area)-1][ind_x2][1]))[0:2]
				x2x1 /= np.linalg.norm(x2x1)
				x2x3 /= np.linalg.norm(x2x3)
				x2xp /= np.linalg.norm(x2xp)

				bisectr = -(x2x1 + x2x3)/2

				#res1 = np.linalg.inv(np.c_[x2x1.reshape((-1, 1)), bisectr.reshape((-1, 1))]) @ x2xp
				#res2 = np.linalg.inv(np.c_[x2x3.reshape((-1, 1)), bisectr.reshape((-1, 1))]) @ x2xp
				if np.all(np.linalg.inv(np.c_[x2x1.reshape((-1, 1)), bisectr.reshape((-1, 1))]) @ x2xp > 0):
					insert = ind_x2
				else:
					insert = ind_x3

		self.area[len(self.area)-1].insert(insert, added_point)

		self.set_closest_point(added_point, 0)
		self.master.configure(cursor="fleur")

		self.redraw_image()

	def refine_closest_point(self, x, y):
		closest_point = None
		closest_distance = 1e100
		#print(len(self.area))
		for point in self.area[len(self.area)-1]:
			point[0] = self.empty_im
			canvas_coords = self.mat_affine @ point[1]
			current_distance = (canvas_coords[0] - x) ** 2 + (canvas_coords[1] - y) ** 2
			if current_distance < closest_distance:
				closest_distance = current_distance
				closest_point = point
		self.set_closest_point(closest_point, closest_distance)
		if self.closest_distance < 20 ** 2:
			self.master.configure(cursor="fleur")
			self.closest_point[0] = self.circled_img
		else:
			self.master.configure(cursor="arrow")

	def delete_point(self, event):
		if self.closest_distance <= 20**2:
			self.area[len(self.area)-1].remove(self.closest_point)
		self.refine_closest_point(event.x, event.y)
		self.redraw_image()

	def set_closest_point(self, point, distance):
		self.prev_closest_point = self.closest_point
		self.prev_closest_distance = self.closest_distance
		self.closest_point = point
		self.closest_distance = distance

	def next(self):
		if len(self.area[len(self.area) - 1]) < 3:
			return
		mask = Image.new("L", self.pil_image.size, 0)
		draw = ImageDraw.Draw(mask)
		draw.polygon([tuple(point[1][:-1]) for point in self.area[len(self.area) - 1]], fill=255, outline=None)
		mask_bbox = mask.getbbox()
		black = Image.new("RGBA", self.pil_image.size, 0)
		crop_mask = mask.crop(mask_bbox)
		crop_black = black.crop(mask_bbox)
		crop_origin = self.pil_image.crop(mask_bbox)
		self.images.append(self.crop(self.subfields[-1]))
		self.area.append([])
		self.subfields.append([])
	def transBg(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
		morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

		roi, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		mask = np.zeros(img.shape, img.dtype)

		cv2.fillPoly(mask, roi, (255,) * img.shape[2], )

		masked_image = cv2.bitwise_and(img, mask)

		return masked_image

	def crop(self, points):
		self.subfields.append(points)

		mask = np.zeros(np.array(self.pil_image.convert('RGB')).shape[0:2], dtype=np.uint16)
		points = np.array(points)

		# print(points)

		rect = cv2.boundingRect(points)
		x, y, w, h = rect
		croped = np.array(self.pil_image.convert('RGB'))[y:y + h, x:x + w].copy()

		pts = points - points.min(axis=0)

		mask = np.zeros(croped.shape[:2], np.uint8)
		cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

		dst = cv2.bitwise_and(croped, croped, mask=mask)

		bg = np.ones_like(croped, np.uint8) * 255
		alpha = np.sum(self.transBg(dst), axis=-1) > 0
		alpha = np.uint8(alpha * 255)
		result = np.dstack((self.transBg(dst), alpha))
		return result

	def explore(self):

		rt = RotateImageInterface(self.images, 1000, 1000, "Анализ", self.master)


class ImageViewer:
	def __init__(self, images):
		self.images = images


if __name__ == "__main__":
	root = tk.Tk()
	app = Application(master=root)
	app.mainloop()
