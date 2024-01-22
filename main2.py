import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage import gaussian_filter

def elastic_twist_local(image, max_angle, center_x, center_y, radius, distortion_radius):
    height, width = image.shape[:2]

    deformation_map_x = np.zeros_like(image, dtype=np.float32)
    deformation_map_y = np.zeros_like(image, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            max_distance = radius

            if distance < radius:
                # 扭曲区域内进行扭曲
                angle = (1 - np.power(distance / max_distance, 2)) * max_angle
                angle_rad = np.deg2rad(angle)
                stretch = 1.0

                # 添加局部模糊效果
                if distance > (radius - distortion_radius):
                    blur_radius = (distance - (radius - distortion_radius)) / distortion_radius
                    image[y, x] = gaussian_filter(image[y, x], sigma=blur_radius)

                deformation_map_x[y, x] = center_x + (x - center_x) * np.cos(angle_rad) * stretch - (y - center_y) * np.sin(angle_rad) * stretch
                deformation_map_y[y, x] = center_y + (x - center_x) * np.sin(angle_rad) * stretch + (y - center_y) * np.cos(angle_rad) * stretch
            else:
                deformation_map_x[y, x] = x
                deformation_map_y[y, x] = y

    twisted_image = cv2.remap(image, deformation_map_x, deformation_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return twisted_image

def save_image(image):
    save_path = filedialog.asksaveasfilename(defaultextension=".png")
    if save_path:
        cv2.imwrite(save_path, image)

def select_image_and_apply_twist():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        max_angle = 40  # 最大扭曲角度
        center_x, center_y = image.shape[1] // 2, image.shape[0] * 3 // 4  # 中心点
        radius = min(center_x, center_y) // 4  # 感兴趣区域的半径
        distortion_radius = radius // 4  # 模糊扭曲的半径

        global twisted_image
        twisted_image = elastic_twist_local(image, max_angle, center_x, center_y, radius, distortion_radius)
        cv2.imshow("Twisted Image with Local Distortion", twisted_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

root = tk.Tk()
root.title("Local Elastic Twist Transformation with Local Distortion")

select_button = tk.Button(root, text="Select Image", command=select_image_and_apply_twist)
select_button.pack()

save_button = tk.Button(root, text="Save Image", command=lambda: save_image(twisted_image))
save_button.pack()

root.mainloop()

