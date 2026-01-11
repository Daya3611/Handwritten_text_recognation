import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import StringLookup
try:
    from tensorflow.keras.saving import register_keras_serializable
except ImportError:
    from tensorflow.keras.utils import register_keras_serializable
import numpy as np
import cv2
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from word_detector import detect, prepare_img, sort_multiline
from transformers import pipeline

# Set page config
st.set_page_config(page_title="Handwritten Text Recognition", layout="wide")

# Constants
KERNEL_SIZE = 25
SIGMA = 11
THETA = 7
MIN_AREA = 100
IMG_HEIGHT = 1000
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32
MAX_LEN = 21
BATCH_SIZE = 64

# --- Helper Functions ---

# Define CTCLayer (Must handle serialization if training, but for inference we just need it to load)
@register_keras_serializable()
class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred
    
    # Need get_config for saving/loading if custom layer logic requires it, 
    # but since we are loading a pre-saved model, this init should be enough 
    # if we pass it to custom_objects.

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check the amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )
    
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image_from_array(image_array):
    """
    Preprocess a numpy array image (H, W, C) or (H, W) for the model.
    """
    # Convert to tensor
    image = tf.convert_to_tensor(image_array)
    
    # Ensure it has a channel dimension
    if len(image.shape) == 2:
        image = tf.expand_dims(image, -1)
    
    # In app.py: image = tf.image.decode_png(image, 1) -> implies Grayscale
    # If input is RGB/BGR, convert to Grayscale
    if image.shape[-1] == 3:
        image = tf.image.rgb_to_grayscale(image)
        
    image = distortion_free_resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Custom LSTM to handle deprecated 'time_major' argument
@register_keras_serializable()
class CustomLSTM(keras.layers.LSTM):
    @classmethod
    def from_config(cls, config):
        if 'time_major' in config:
            del config['time_major']
        return super().from_config(config)

@st.cache_resource
def load_resources():
    # Load Vocabulary
    with open("./characters", "rb") as fp:
        vocab = pickle.load(fp)
    
    char_to_num = StringLookup(vocabulary=vocab, mask_token=None)
    num_to_chars = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
    
    # Load Model
    # Explicitly map standard layers that might be missing in registry for some versions
    custom_objects = {
        "CTCLayer": CTCLayer,
        "LSTM": CustomLSTM,
        "Bidirectional": keras.layers.Bidirectional
    }
    reconstructed_model = keras.models.load_model("./ocr_model_50_epoch.h5", custom_objects=custom_objects)
    
    # Create prediction model (extracting inference part)
    # Find the image input tensor. The model has two inputs: 'image' and 'label'.
    # We need 'image' for inference.
    image_input = None
    for input_tensor in reconstructed_model.inputs:
        # Check name or shape. Image shape is (None, 128, 32, 1) or similar.
        if "image" in input_tensor.name:
            image_input = input_tensor
            break
            
    if image_input is None:
        # Fallback to the first input if name doesn't match
        image_input = reconstructed_model.inputs[0]

    prediction_model = keras.models.Model(
        inputs=image_input, 
        outputs=reconstructed_model.get_layer(name="dense2").output
    )

    # Load Correction Model (Pipeline)
    # Using oliverguhr/spelling-correction-english-base for robust OCR correction
    spell_corrector = pipeline(
        "text2text-generation",
        model="oliverguhr/spelling-correction-english-base"
    )

    return prediction_model, char_to_num, num_to_chars, spell_corrector

def lm_correct(text, spell_corrector):
    if len(text.strip()) < 5:
        return text

    # The model expects a string input
    try:
        result = spell_corrector(
            text,
            max_length=256,
            clean_up_tokenization_spaces=True
        )
        return result[0]["generated_text"]
    except Exception as e:
        # Fallback if something goes wrong (e.g. empty string after strip check passed)
        return text

def decode_batch_predictions(pred, num_to_chars):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :MAX_LEN
    ]

    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_chars(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

# --- Main App ---

st.title("📝 Handwritten Text Recognition")
st.markdown("Upload an image of handwritten text (full page or line) to detect and recognize the content.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    # Load resources
    try:
        model, char_to_num, num_to_chars, spell_corrector = load_resources()
    except Exception as e:
        st.error(f"Error loading model resources: {e}")
        st.stop()

    # Read image
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)
        st.write("Processing...")

    # Detection / Segmentation
    try:
        # Prepare image for segmentation (Grayscale, Resized)
        # word_detector expects BGR or Grayscale. standard cv2.imread is BGR. 
        # Image.open is RGB. We should convert to BGR for compatibility or just Grayscale.
        # prepare_img converts to grayscale internally if needed.
        
        # Convert RGB to BGR for cv2 compatibility just in case
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        processed_img = prepare_img(img_bgr, IMG_HEIGHT)
        
        detections = detect(processed_img,
                            kernel_size=KERNEL_SIZE,
                            sigma=SIGMA,
                            theta=THETA,
                            min_area=MIN_AREA)
        
        lines = sort_multiline(detections)
        
    except Exception as e:
        st.error(f"Error during segmentation: {e}")
        st.stop()
        
    # Recognition
    full_text = []
    line_images = []
    
    with col2:
        st.subheader("Recognized Text")
        if not lines:
            st.warning("No text detected.")
        else:
            progress_bar = st.progress(0)
            total_lines = len(lines)
            
            for line_idx, line in enumerate(lines):
                line_text = []
                # Collect word images for this line
                word_imgs = []
                word_bboxes = []
                
                for word_det in line:
                    crop = word_det.img # This is the cropped word image from detector
                    
                    # Preprocess for model
                    # Crop is grayscale numpy array from cv2 (H, W)
                    # Helper needs (H, W, C) or handle simple 2D
                    processed_crop = preprocess_image_from_array(crop)
                    word_imgs.append(processed_crop)
                    word_bboxes.append(word_det.bbox)
                
                if not word_imgs:
                    continue
                    
                # Batch prediction for the line
                batch_images = tf.stack(word_imgs)
                preds = model.predict(batch_images, verbose=0)
                pred_texts = decode_batch_predictions(preds, num_to_chars)
                
                # Join words to form line text
                current_line_str = " ".join(pred_texts)
                
                # Apply Correction
                corrected_line = lm_correct(current_line_str, spell_corrector)
                
                full_text.append(corrected_line)
                st.write(f"Line {line_idx+1}: **{corrected_line}** (Raw: *{current_line_str}*)")
                
                progress_bar.progress((line_idx + 1) / total_lines)
    
    # Show Full Text
    st.divider()
    st.header("Final Result")
    st.text_area("Transcribed Text", value="\n".join(full_text), height=200)

    # Optional: Visualization of detected boxes
    if st.checkbox("Show Detected Word Boxes"):
        plt_img = processed_img.copy()
        plt.figure(figsize=(10, 10))
        plt.imshow(plt_img, cmap='gray')
        for line_idx, line in enumerate(lines):
            for word_idx, det in enumerate(line):
                x, y, w, h = det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h
                plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
                plt.text(x, y - 5, f'{line_idx}.{word_idx}', color='blue', fontsize=8)
        
        st.pyplot(plt)
