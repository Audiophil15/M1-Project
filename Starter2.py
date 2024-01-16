import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


def bilinear_interpolate(image, x, y):
    # Get the dimensions of the image
    height, width, channels = image.shape
    # Calculate the coordinates of the four surrounding pixels
    x1, y1 = int(x), int(y)
    x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)
    # Calculate the differences
    dx, dy = x - x1, y - y1
    # Interpolate
    interpolated = np.zeros(channels)
    for c in range(channels):
        interpolated[c] = (image[y1, x1, c] * (1 - dx) * (1 - dy) +
                           image[y1, x2, c] * dx * (1 - dy) +
                           image[y2, x1, c] * (1 - dx) * dy +
                           image[y2, x2, c] * dx * dy)
    return interpolated


def rotate_image_with_bilinear_interpolation(image_path, angle):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    # Calculate the center
    center_x, center_y = width / 2, height / 2
    # Create the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    # Create an empty image for the output
    rotated_image = np.zeros_like(image)
    # Perform the rotation
    for y in range(height):
        for x in range(width):
            # Apply the rotation matrix
            new_x, new_y = np.dot(rotation_matrix, np.array([x, y, 1]))
            if 0 <= new_x < width and 0 <= new_y < height:
                rotated_image[y, x, :] = bilinear_interpolate(image, new_x, new_y)
    return rotated_image


def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()  # Show the file dialog
    return file_path


# Main execution
image_path = select_image()
print("Selected Image Path:", image_path)  # Add this line for debugging
if image_path:  # Check if a file was selected
    angle = 315  # Set the angle for rotation
    rotated = rotate_image_with_bilinear_interpolation(image_path, angle)
    cv2.imshow('Rotated Image', rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# save
output_path = r"C:\Users\14935\Desktop\output.jpg"  # ʹ��ԭʼ�ַ�����ʾ·��
cv2.imwrite(output_path, rotated)
print("Rotated image saved at:", output_path)
