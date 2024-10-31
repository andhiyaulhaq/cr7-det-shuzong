# Cold-Rolled Steel Strips Surface Defects Detection

### Overview

This project aims to enhance the automated detection of surface defects in cold-rolled steel strips, which are essential for various industrial applications, by utilizing CR7-DET Dataset. This project employs the YOLO11 algorithm and Roboflow inference to develop a desktop application focused specifically on efficient defect identification. Additionally, two separate face detection programs are provided, serving as benchmarks for evaluating the performance of the surface defect detection application. By addressing the limitations of traditional manual inspection methods, which are labor-intensive and error-prone, this initiative seeks to streamline the defect detection process, enabling proactive quality assurance and improved operational efficiency.

### References

- [Scientific Article on CR7-DET Dataset](https://www.sciencedirect.com/science/article/pii/S0952197624014830)
- [Roboflow Inference Guide](https://inference.roboflow.com)
- [Roboflow Inference for Face Detection](https://blog.roboflow.com/inference-python/)

### Dataset

The dataset is located in the `rolled_data` folder (source: [CR7-DET GitHub](https://github.com/jsq0903/CR7-DET))

### Requirements Note

Inference package requires Python 3.8 ≤ version ≤ 3.11

### Setup and Usage

1. Clone this repository.
2. Create a virtual environment (use Python 3.8 ≤ version ≤ 3.11):

   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:

   - On macOS/Linux:

     ```bash
     source .venv/bin/activate
     ```

   - On Windows:

     ```cmd
     .venv\Scripts\activate
     ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the desired script:

   ```bash
   python steel-image-detail.py
   ```

### Script Description

- `face-image.py` : Detects faces from a selected local image. This script allows the user to select an image file from their system, performs face detection, and displays the results.
- `face-webcam.py` : Detects faces in real-time using a webcam. It continuously processes webcam video frames for face detection and displays the results live.
- `steel-image.py` : Detects defects on a selected steel sheet image. This script identifies defects on the steel sheet and overlays defect names with confidence percentages directly onto the image.
- `steel-image-detail.py` : Processes an image of a steel sheet to identify defects, similar to `steel-image.py`. However, defect details (including confidence percentages) are organized in a table instead of directly on the image. Each defect has a unique ID, displayed on the processed image.

### License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/andhiyaulhaq/cr7-shuzong/blob/main/LICENSE) file for more details.
