import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
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

# Global variables to hold the processed image and defect details
processed_image = None
defect_details = []


def open_and_detect_image():
    global processed_image, defect_details  # Use global variables to hold processed image and defect details
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    if file_path:
        start_time = time.time()

        # Load image with OpenCV
        frame = cv2.imread(file_path)

        # Inference image to find defects
        results = model.infer(image=frame, confidence=0.5, iou_threshold=0.5)

        # Clear previous data in the defect table
        for item in defect_table.get_children():
            defect_table.delete(item)

        # Clear previous defect details
        defect_details.clear()

        # Plot image with bounding box and ID (using OpenCV)
        if results[0].predictions:
            for idx, prediction in enumerate(results[0].predictions, start=1):
                x_center = int(prediction.x)
                y_center = int(prediction.y)
                width = int(prediction.width)
                height = int(prediction.height)
                class_name = prediction.class_name
                confidence = prediction.confidence * 100

                # Save details for the defect, including the ID
                defect_details.append(
                    (
                        idx,  # Include ID
                        class_name,
                        f"{confidence:.1f}%",  # Store confidence as string
                        x_center,
                        y_center,
                        width,
                        height,
                    )
                )

                # Calculate top-left and bottom-right corners
                x0 = x_center - width // 2
                y0 = y_center - height // 2
                x1 = x_center + width // 2
                y1 = y_center + height // 2

                # Draw bounding box with ID on the displayed image (red color)
                label_color = (238, 0, 0)  # Red color
                cv2.rectangle(frame, (x0, y0), (x1, y1), label_color, 1)
                cv2.putText(
                    frame,
                    str(idx),  # Display the ID
                    (x0, y0 - 5 if y0 - 5 > 0 else y1 + 5),
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
        processed_image = detected_image_rgb

        # Display the image with annotations
        image_label.config(image=photo)
        image_label.image = photo

        end_time = time.time()
        processing_time = (end_time - start_time) * 1000

        # Update detailed defect info
        for detail in defect_details:
            defect_table.insert(
                "",
                "end",
                values=(
                    detail[0],
                    detail[1],
                    detail[2],
                ),  # Display ID, Class, and Confidence
            )

        # Update total processing time in the Text widget
        processing_textbox.configure(state="normal")
        processing_textbox.delete("1.0", tk.END)
        processing_textbox.insert(
            tk.END, f"Total processing time: {processing_time:.1f}ms"
        )
        processing_textbox.configure(state="disabled")

        # Enable save button after processing
        save_image_button.config(state="normal")


def save_image():
    global processed_image, defect_details
    if processed_image is not None:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg;*.jpeg"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            # Save the processed image with defect names and confidence
            save_image_with_defects(processed_image, defect_details, file_path)


def save_image_with_defects(image, details, file_path):
    # Create a copy of the image to draw labels on
    image_copy = image.copy()
    for detail in details:
        id_num, class_name, confidence, x_center, y_center, width, height = (
            detail  # Unpack correctly
        )

        # Calculate top-left and bottom-right corners
        x0 = x_center - width // 2
        y0 = y_center - height // 2
        x1 = x_center + width // 2
        y1 = y_center + height // 2

        # Draw bounding box and label on saved image
        label_color = (0, 0, 238)  # Use blue color
        cv2.rectangle(image_copy, (x0, y0), (x1, y1), label_color, 1)
        label_text = (
            f"{class_name} {confidence}"  # Include only the class name and confidence
        )
        cv2.putText(
            image_copy,
            label_text,
            (x0, y0 - 5 if y0 - 5 > 0 else y1 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            label_color,
            1,
        )

    # Save the modified image
    Image.fromarray(image_copy).save(file_path)


# Create the main window
root = tk.Tk()
root.title("Roboflow Defect Detection Viewer")
root.geometry("400x700")

# Create a label for the placeholder
placeholder_image = Image.new("RGB", (300, 300), "grey")
placeholder_photo = ImageTk.PhotoImage(placeholder_image)
image_label = tk.Label(root, image=placeholder_photo)
image_label.pack(pady=10)

# Create a table for detailed defect information
defect_table = ttk.Treeview(
    root, columns=("ID", "Class", "Confidence"), show="headings", height=8
)
defect_table.column("ID", width=50, anchor="center")
defect_table.column("Class", width=150, anchor="center")
defect_table.column("Confidence", width=100, anchor="center")
defect_table.heading("ID", text="ID")
defect_table.heading("Class", text="Class")
defect_table.heading("Confidence", text="Confidence")
defect_table.pack(pady=10)

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
processing_textbox.insert(tk.END, "Total processing time: ")
processing_textbox.configure(state="disabled")

# Create a button to open and detect objects in the image
open_image_button = tk.Button(
    root, text="Select and Detect Image", command=open_and_detect_image
)
open_image_button.pack(pady=10)

# Create a button to save the processed image, disabled initially
save_image_button = tk.Button(
    root, text="Save Image", command=save_image, state="disabled"
)
save_image_button.pack(pady=10)

# Run the application
root.mainloop()
