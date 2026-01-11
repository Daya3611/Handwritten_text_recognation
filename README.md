# Handwritten Text Detection and Recognition

This project implements a full pipeline for detecting and recognizing handwritten text from page images. It uses a computer vision-based approach for word segmentation and a Deep Learning model (CRNN with CTC loss) for text recognition.

## Project Structure

- `app.py`: Main entry point for the application. Handles image loading, segmentation, and recognition inference.
- `word_detector/`: Contains the logic for word segmentation.
  - `__init__.py`: Implements the scale space technique for word segmentation and line clustering.
- `characters`: Pickle file containing the vocabulary for the recognition model.
- `ocr_model_50_epoch.h5`: Pre-trained Keras model for handwriting recognition.
- `data/`: Directory for input images.
- `test_images/`: Directory where segmented word images are saved.

## Usage

1.  **Install Dependencies**: Ensure you have the required packages installed (TensorFlow, OpenCV, NumPy, Matplotlib, etc.).
2.  **Run the Application**:
    ```bash
    python app.py --data ./data/page
    ```
    Arguments:
    - `--data`: Path to the directory containing page images.
    - `--kernel_size`: Size of the filter kernel for segmentation (default: 25).
    - `--sigma`: Standard deviation for Gaussian filter (default: 11).
    - `--theta`: Aspect ratio factor for the filter (default: 7).
    - `--min_area`: Minimum area for a contour to be considered a word (default: 100).
    - `--img_height`: Height to resize input images to before processing (default: 1000).

## Detailed System Prompt

Use the following detailed prompt to replicate or understand the logic of this project in other contexts. This prompt encapsulates the core algorithmic strategies used for segmentation and recognition.

### Prompt: Handwritten Text Extraction and Recognition System Logic

**Objective:**
Develop or analyze a system that takes a full-page image of handwritten text, segments it into individual lines and words, and then recognizes the text within each word image.

**1. Word Segmentation Logic (Computer Vision approach)**
The segmentation module uses a localized scale-space technique (based on R. Manmatha's paper) to identify word boundaries.

*   **Preprocessing:**
    *   Convert the input image to grayscale.
    *   Resize the image to a fixed height (e.g., 1000px) while maintaining aspect ratio to ensure consistent feature sizes.

*   **Anisotropic Filtering:**
    *   Create a custom Gaussian filter kernel defined by `kernel_size` (e.g., 25), `sigma` (e.g., 11), and `theta` (e.g., 7).
    *   `theta` represents the average width/height ratio of words, elongating the kernel horizontally to match word shapes.
    *   Apply this `filter2D` to the image. This blurs the image in a way that connects characters within a word while keeping separate words distinct.

*   **Thresholding & Extraction:**
    *   Apply Otsu's binary thresholding to the filtered image to create a binary mask.
    *   Find contours (`cv2.findContours`) on this mask.
    *   Filter out small noise contours based on `min_area` (e.g., 100 pixels).
    *   Compute the bounding box (`x, y, w, h`) for each valid contour.

*   **Line Clustering:**
    *   Use DBSCAN clustering to group word bounding boxes into lines.
    *   Metric: Jaccard distance based on the vertical (y-axis) overlap of bounding boxes.
    *   Sort lines vertically (top to bottom) and words within each line horizontally (left to right).

**2. Text Recognition Logic (Deep Learning CRNN)**
The recognition module processes individual word crops using a Convolutional Recurrent Neural Network (CRNN) trained with CTC (Connectionist Temporal Classification) loss.

*   **Input Preprocessing:**
    *   Resize word images to a fixed height (32px) and scale width proportionally.
    *   Pad images to a fixed size (128x32) using specific padding rules (center or edge padding) to handle variable word lengths.
    *   Normalize pixel values to [0, 1].

*   **Model Architecture:**
    *   **CNN Encoder:**
        *   Block 1: Conv2D (32 filters, 3x3) -> MaxPool (2x2).
        *   Block 2: Conv2D (64 filters, 3x3) -> MaxPool (2x2).
    *   **Reshape:** Flatten spatial dimensions (width/4, height/4 * filters) to prepare for the sequence model.
    *   **Dense Projection:** Dense layer (64 units) + Dropout to map features.
    *   **RNN Decoder:**
        *   Bidirectional LSTM (128 units).
        *   Bidirectional LSTM (64 units).
    *   **Output Layer:** Dense layer with Softmax activation. Output size = Vocabulary size + 1 (CTC blank token).

*   **Decoding:**
    *   Use `CTC Greedy Decode` to convert the frame-wise probability outputs from the RNN into the final character sequence.
    *   Detailed logic involves mapping the integer indices back to characters using a predefined vocabulary mapping.

**3. Workflow Summary**
1.  Load page image.
2.  **Segment** page into sorted lines and words using the CV key-point/filtering method.
3.  **Crop** each word image.
4.  **Preprocess** crop (resize/pad) for the model.
5.  **Infer** text using the loaded CRNN model.
6.  **Reconstruct** full text by joining recognized words.
