import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time
from inference.models.utils import get_roboflow_model

# Roboflow model details
model_name = "face-detection-mik1i"
model_version = "18"
api_key = "kpZmuqOmaWfbgjr8KNW0"  # Replace with your actual API key

# Load the Roboflow face detection model
model = get_roboflow_model(model_id=f"{model_name}/{model_version}", api_key=api_key)

# Global variables
cap = None
running = False
paused = False


def start_camera():
    global cap, running, paused
    cap = cv2.VideoCapture(0)  # Access the webcam
    running = True
    pause_button.config(state=tk.NORMAL)
    start_button.config(state=tk.DISABLED)
    update_frame()  # Start updating frames


def update_frame():
    global cap, running, paused
    if running:
        ret, frame = cap.read()  # Read a frame from the camera
        if ret:
            start_time = time.time()  # Start time before processing

            # Perform inference with Roboflow model
            results = model.infer(image=frame, confidence=0.5, iou_threshold=0.5)
            if results[0].predictions:
                for prediction in results[0].predictions:
                    x_center, y_center = int(prediction.x), int(prediction.y)
                    width, height = int(prediction.width), int(prediction.height)
                    x0, y0 = x_center - width // 2, y_center - height // 2
                    x1, y1 = x_center + width // 2, y_center + height // 2
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 2)
                    cv2.putText(
                        frame,
                        "Face",
                        (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            # Convert BGR to RGB for Tkinter display
            detected_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_image = Image.fromarray(detected_image_rgb)
            annotated_image.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(annotated_image)

            # Display annotated image
            image_label.config(image=photo)
            image_label.image = photo

            # Display processing time
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # in milliseconds
            processing_text = f"Total processing time: {processing_time:.1f}ms"
            processing_textbox.configure(state="normal")
            processing_textbox.delete("1.0", tk.END)
            processing_textbox.insert(tk.END, processing_text)
            processing_textbox.configure(state="disabled")

            # Show FPS
            fps = 1 / (end_time - start_time)
            fps_label.config(text=f"FPS: {fps:.2f}")

        if not paused:
            image_label.after(1, update_frame)  # Continue updating frames if not paused


def pause_camera():
    global paused
    paused = not paused
    pause_button.config(text="Resume" if paused else "Pause")
    if not paused:
        update_frame()  # Resume updating frames


# Create Tkinter window
root = tk.Tk()
root.title("Roboflow Face Detection Viewer")
root.geometry("600x700")

# Start camera button
start_button = tk.Button(root, text="Start Camera", command=start_camera)
start_button.pack(pady=10)

# Pause camera button
pause_button = tk.Button(root, text="Pause", command=pause_camera, state=tk.DISABLED)
pause_button.pack(pady=10)

# Image display label
image_label = tk.Label(root)
image_label.pack(pady=10)

# FPS display label
fps_label = tk.Label(root, text="FPS: 0.00")
fps_label.pack(pady=5)

# Processing time display
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
processing_textbox.configure(state="disabled")

# Run the Tkinter main loop
root.mainloop()

# Release camera on exit
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
