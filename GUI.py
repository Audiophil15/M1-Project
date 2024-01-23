import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk, ImageFilter

def choose_image():
    global original_image, image_display
    filename = filedialog.askopenfilename()
    if filename:
        original_image = Image.open(filename)
        photo = ImageTk.PhotoImage(original_image)
        image_display.config(image=photo)
        image_display.image = photo
        image_display.pack()

def apply_corruption(option):
    global original_image, image_display
    if original_image:
        if option == 'warps':
            processed_image = original_image  # Placeholder for actual warping
        elif option == 'rotate':
            processed_image = original_image.rotate(90)  # Example rotation
        elif option == 'blurring':
            processed_image = original_image.filter(ImageFilter.BLUR)
        elif option == 'pixels permute':
            processed_image = original_image  # Placeholder for actual permutation

        photo = ImageTk.PhotoImage(processed_image)
        image_display.config(image=photo)
        image_display.image = photo  # keep a reference so it's not garbage collected
    else:
        messagebox.showerror("Error", "No image selected!")

def save_image():
    global original_image
    if original_image:
        save_filename = filedialog.asksaveasfilename(defaultextension=".png")
        if save_filename:
            original_image.save(save_filename)
    else:
        messagebox.showerror("Error", "No processed image to save!")

# Create the main window
root = tk.Tk()
root.title("Project")

# Original and processed image display
image_display = tk.Label(root)
image_display.pack()

# Choose image button
choose_button = tk.Button(root, text="Choose image", command=choose_image)
choose_button.pack()

# Corruption options
corruption_frame = tk.LabelFrame(root, text="Apply corruption", padx=10, pady=10)
corruption_frame.pack(padx=10, pady=10)

# Define corruption options
corruption_options = ['warps', 'rotate', 'blurring', 'pixels permute']

for option in corruption_options:
    button = tk.Button(corruption_frame, text=option, command=lambda opt=option: apply_corruption(opt))
    button.pack(anchor='w')

# Save image button
save_button = tk.Button(root, text="Save image", command=save_image)
save_button.pack()

# Start the GUI loop
original_image = None
root.mainloop()

