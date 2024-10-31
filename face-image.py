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

API_KEY = os.getenv("API_KEY")  # Replace with your actual API key

# Load the Roboflow face detection model
model = get_roboflow_model(model_id=f"{model_name}/{model_version}", api_key=API_KEY)


def open_and_detect_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    if file_path:
        start_time = time.time()  # Start time before processing

        # Load image with OpenCV
        frame = cv2.imread(file_path)

        # Inference image to find faces
        results = model.infer(image=frame, confidence=0.5, iou_threshold=0.5)

        # Plot image with face bounding box (using OpenCV)
        if results[0].predictions:
            for prediction in results[0].predictions:
                x_center = int(prediction.x)
                y_center = int(prediction.y)
                width = int(prediction.width)
                height = int(prediction.height)

                # Calculate top-left and bottom-right corners from center, width, and height
                x0 = x_center - width // 2
                y0 = y_center - height // 2
                x1 = x_center + width // 2
                y1 = y_center + height // 2

                cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 10)
                cv2.putText(
                    frame,
                    "Face",
                    (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    2,
                )

        # Convert BGR to RGB for Tkinter
        detected_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_image = Image.fromarray(detected_image_rgb)
        annotated_image.thumbnail(
            (400, 300), Image.Resampling.LANCZOS
        )  # Resize to fit window
        photo = ImageTk.PhotoImage(annotated_image)

        # Display the image with annotations
        image_label.config(image=photo)
        image_label.image = photo  # Keep reference to avoid garbage collection

        # End time after processing is done
        end_time = time.time()  # Record end time
        processing_time = (end_time - start_time) * 1000  # Convert to milliseconds

        # Summary of detections
        detection_count = len(results[0].predictions)
        summary_text = (
            f"image 1/1 {file_path}: {frame.shape[1]}x{frame.shape[0]} "
            f"{detection_count} detections"
        )

        # Enable textbox to insert text and then disable it
        summary_textbox.configure(state="normal")
        summary_textbox.delete("1.0", tk.END)
        summary_textbox.insert(tk.END, summary_text)
        summary_textbox.configure(state="disabled")

        # Display total processing time
        processing_text = f"Total processing time: {processing_time:.1f}ms"
        processing_textbox.configure(state="normal")
        processing_textbox.delete("1.0", tk.END)
        processing_textbox.insert(tk.END, processing_text)
        processing_textbox.configure(state="disabled")


# Create the main window
root = tk.Tk()
root.title("Roboflow Face Detection Viewer")
root.geometry("500x500")  # Set fixed size to 600x700

# Create a button to open and detect objects in the image
open_image_button = tk.Button(
    root, text="Select and Detect Image", command=open_and_detect_image
)
open_image_button.pack(pady=10)

# Create a label to display the annotated image
image_label = tk.Label(root)
image_label.pack(pady=10)

# Create a text box for the detection summary, configured to look like a label
summary_textbox = tk.Text(
    root,
    height=4,
    width=60,
    wrap="word",
    font=("Arial", 10),
    bg=root.cget("bg"),
    relief="flat",
)
summary_textbox.pack(pady=5)
summary_textbox.configure(state="disabled")  # Disable typing initially

# Create a text box for the processing time
processing_textbox = tk.Text(
    root,
    height=2,
    width=60,
    wrap="word",
    font=("Arial", 10),
    bg=root.cget("bg"),
    relief="flat",
)
processing_textbox.pack(pady=5)
processing_textbox.configure(state="disabled")  # Disable typing initially

# Run the application
root.mainloop()
