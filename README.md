# CR7-DET Dataset

reference :

- https://www.sciencedirect.com/science/article/pii/S0952197624014830
- https://blog.roboflow.com/inference-python/
- https://inference.roboflow.com

dataset : in `rolled data` folder (source: https://github.com/jsq0903/CR7-DET)

### Note

Inference package requires Python 3.8 ≤ version ≤ 3.11

### Setup and Usage

1. Clone this repository.
2. Create a virtual environment: `python -m venv .venv` (use Python 3.8 ≤ version ≤ 3.11).
3. Activate the virtual environment:
   - On macOS/Linux: `source .venv/bin/activate`
   - On Windows: `.venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the script: `python <script>` (e.g., `python steel-image-detail.py`)

### Description

- `face-image.py` : This script is designed for detecting faces in a local image file chosen by the user. It allows the user to select an image from their system, processes the image to identify faces, and displays the detection results.
- `face-webcam.py` : This script enables real-time face detection using the webcam. It captures video frames from the webcam, applies face detection on each frame, and displays the detection results continuously.
- `steel-image.py` : This script detects defects in a steel sheet image selected by the user. The defects, along with their confidence percentages, are displayed directly on the image itself.
- `steel-image-detail.py` : Similar to `steel-image.py`, this script processes an image of a steel sheet to identify defects. But, instead of displaying details directly on the image, it organizes the defect information—along with confidence percentages—in a table format. Each detected defect is also assigned a unique ID, which is shown on the processed image for easy reference.
