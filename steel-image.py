import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from inference.models.utils import get_roboflow_model
import time
from dotenv import load_dotenv
import os

load_dotenv()

# Roboflow model details
model_name = "cr7-det-shuzong-dataset"
model_version = "2"

API_KEY = os.getenv("API_KEY")

# Get Roboflow defect model
model = get_roboflow_model(
    model_id=f"{model_name}/{model_version}",
    api_key=API_KEY,
)

# Global variable to hold the processed image
processed_image = None


def open_and_detect_image():
    global processed_image  # Use global variable to hold processed image
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    if file_path:
        start_time = time.time()

        # Load image with OpenCV
        frame = cv2.imread(file_path)

        # Inference image to find defects
        results = model.infer(image=frame, confidence=0.5, iou_threshold=0.5)

        # Clear previous defect details
        defect_info = []  # List to store defect info

        # Plot image with bounding box and ID (using OpenCV)
        if results[0].predictions:
            for idx, prediction in enumerate(results[0].predictions, start=1):
                x_center = int(prediction.x)
                y_center = int(prediction.y)
                width = int(prediction.width)
                height = int(prediction.height)
                class_name = prediction.class_name
                confidence = prediction.confidence * 100

                # Append defect details to the list
                defect_info.append(f"{class_name}: {confidence:.1f}%")

                # Calculate top-left and bottom-right corners
                x0 = x_center - width // 2
                y0 = y_center - height // 2
                x1 = x_center + width // 2
                y1 = y_center + height // 2

                # Display defect name with confidence percentage
                label_text = f"{class_name} {confidence:.1f}%"
                text_y = (
                    y0 - 10 if y0 - 10 > 0 else y1 + 20
                )  # Adjust position if text goes out of bounds

                # Sample the region around the label for brightness calculation
                roi_y_start = max(y0 - 20, 0)  # A few pixels above the bounding box
                roi_y_end = min(y0, frame.shape[0])
                roi_x_start = max(x0, 0)
                roi_x_end = min(x1, frame.shape[1])

                # Calculate average brightness
                roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
                avg_brightness = cv2.mean(roi)[:3]  # Ignore alpha channel if present
                avg_brightness_value = (
                    sum(avg_brightness) / 3
                )  # Grayscale average brightness

                label_color = (
                    (255, 255, 0) if avg_brightness_value < 128 else (100, 100, 0)
                )

                # Draw bounding box with certain thickness
                cv2.rectangle(frame, (x0, y0), (x1, y1), label_color, 1)

                cv2.putText(
                    frame,
                    label_text,  # Use actual class name instead of "Face"
                    (x0, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    label_color,
                    1,
                )

        # Convert BGR to RGB for Tkinter
        detected_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_image = Image.fromarray(detected_image_rgb)
        annotated_image.thumbnail((400, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(annotated_image)

        # Store the processed image for saving later
        processed_image = annotated_image

        # Display the image with annotations
        image_label.config(image=photo)
        image_label.image = photo

        # Enable the save button after processing
        save_image_button.config(state="normal")

        end_time = time.time()
        processing_time = (end_time - start_time) * 1000

        # Update total processing time in the Text widget
        processing_textbox.configure(state="normal")
        processing_textbox.delete("1.0", tk.END)
        processing_textbox.insert(
            tk.END, f"Total processing time: {processing_time:.1f}ms"
        )
        processing_textbox.configure(state="disabled")


def save_image():
    global processed_image  # Access the processed image
    if processed_image:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg;*.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            processed_image.save(
                file_path
            )  # Save the processed image to the selected path


# Create the main window
root = tk.Tk()
root.title("Roboflow Defect Detection Viewer")
root.geometry("400x500")

# Create a label for the placeholder
placeholder_image = Image.new("RGB", (300, 300), "grey")
placeholder_photo = ImageTk.PhotoImage(placeholder_image)
image_label = tk.Label(root, image=placeholder_photo)
image_label.pack(pady=20)

# Create a text box for the processing time with initial text
processing_textbox = tk.Text(
    root,
    height=2,
    width=40,
    wrap="word",
    font=("Arial", 10),
    bg=root.cget("bg"),
    relief="flat",
)
processing_textbox.pack(pady=10)
processing_textbox.insert(tk.END, "Total processing time: ")  # Initial text
processing_textbox.configure(state="disabled")  # Disable editing initially

# Create a button to open and detect objects in the image
open_image_button = tk.Button(
    root, text="Select and Detect Image", command=open_and_detect_image
)
open_image_button.pack(pady=0)

# Create a button to save the processed image, initially disabled
save_image_button = tk.Button(
    root, text="Save Image", command=save_image, state="disabled"
)
save_image_button.pack(pady=10)

# Run the application
root.mainloop()
