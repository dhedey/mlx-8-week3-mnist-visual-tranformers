import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import numpy as np
import torch
import io
import sys
import os
import torchvision
import torchvision.transforms.v2 as v2
import einops
from uuid import uuid4
import streamlit.components.v1 as components

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.inference import get_data_info, select_model, predict_sequence
from model.models import DigitSequenceModel
from model.common import select_device
from model.composite_dataset import BesCombine, CompositeDataset
from model.create_composite_david import composite_image_generator_david


st.set_page_config(page_title="Overlord's Vision", page_icon="👀")

st.title("Overlord's Vision")
st.write("A multi-digit MNIST sequence predictor.")

# Invert the display_names dictionary to map display names back to keys
display_names = {
    "best5by5_scramble": "5 x 5 scrambled",
    "best4by4_var": "4 x 4 on-grid",
    "best2by2": "2 x 2 on-grid (fixed length)",
}
name_to_key = {v: k for k, v in display_names.items()}

# --- Model Selection ---
model_display_name = st.selectbox(
    "Choose a model to test:",
    options=list(name_to_key.keys()),
    index=0,
)
model_key = name_to_key[model_display_name]

@st.cache_resource
def load_model(model_name_key):
    """Loads and caches the selected model."""
    device = select_device()
    model_path = select_model(model_name_key)["model_path"]
    model = DigitSequenceModel.load_for_evaluation(model_path=model_path, device=device)
    return model

def create_tick_marks_image(height, h_patches):
    """Creates a small image with tick marks for grid alignment."""
    width = 30  # A narrow strip for the ticks
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)
    
    cell_height = height / h_patches
    
    for i in range(1, h_patches):
        y = i * cell_height
        draw.line([(0, y), (width, y)], fill="white", width=1)
        
    return img

def create_horizontal_tick_marks_image(width, w_patches):
    """Creates a small image with horizontal tick marks for grid alignment."""
    height = 30  # A narrow strip for the ticks
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)
    
    cell_width = width / w_patches
    
    for i in range(1, w_patches):
        x = i * cell_width
        draw.line([(x, 0), (x, height)], fill="white", width=1)
        
    return img

def create_image_generator(_model, _data_info):
    """Creates an image generator/iterator based on the data info."""
    data_type = _data_info["type"]
    h_patches = _data_info["h_patches"]
    w_patches = _data_info["w_patches"]

    if data_type == "bes":
        p_skip = 0.2 if _data_info["variable_length"] else 0
        dataset = BesCombine(train=False, h_patches=h_patches, w_patches=w_patches, length=100, p_skip=p_skip)
        return iter(dataset)
    elif data_type == "david":
        data_folder = os.path.join(os.path.dirname(__file__), "..", "model", "datasets")
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.RandomResize(28, 40),
            v2.RandomRotation(25),
            v2.RandomResizedCrop(size=28, scale=(28.0/40, 28.0/40)),
        ])
        mnist_ds = torchvision.datasets.MNIST(data_folder, download=True, transform=transform, train=False)
        
        david_generator = composite_image_generator_david(
            image_dataset=mnist_ds,
            output_width=_model.config.encoder.image_width,
            output_height=_model.config.encoder.image_height,
            output_batch_size=1,
            batches_per_epoch=100, # Create a longer-running generator
            line_height_min=16,
            line_height_max=64,
        )
        return david_generator
    else:
        st.error(f"Example generation not supported for type: {data_type}")
        return None

def get_next_example_image(model_key, model, data_info):
    """Gets the next image from a cached generator, creating one if needed."""
    if 'image_generator' not in st.session_state or st.session_state.get('model_key') != model_key:
        st.session_state.image_generator = create_image_generator(model, data_info)
        st.session_state.model_key = model_key

    generator = st.session_state.image_generator
    if generator is None:
        return None
        
    try:
        data_type = data_info["type"]
        if data_type == "bes":
            example_image_tensor, _ = next(generator)
        elif data_type == "david":
            img_batch, _, _ = next(generator)
            example_image_tensor = img_batch[0]
        else:
            return None
        return example_image_tensor
    except StopIteration:
        st.warning("Image generator exhausted. Re-initializing...")
        # Clear the old generator and recursively call to get a new one
        del st.session_state['image_generator']
        return get_next_example_image(model_key, model, data_info)

with st.spinner(f"Loading model: {model_display_name}..."):
    model = load_model(model_key)

data_info = get_data_info(model_key)
image_size = data_info["image_size"]

# --- Drawable Canvas ---
st.write(f"Draw a digit sequence. The model expects an image of size {image_size[1]}x{image_size[0]}. Your drawing will be resized.")

canvas_height = 560
canvas_width = 560

main_col, right_col = st.columns([10, 1])

with main_col:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="freedraw",
        key="canvas",
    )
    # tick mark images removed; grid overlay handles guidance

with right_col:
    pass  # tick mark sidebar removed

# --- Prediction Logic ---
predict_col, example_col = st.columns(2)
with predict_col:
    predict_button = st.button("Predict from Drawing", use_container_width=True)
with example_col:
    example_button = st.button("Get Example Image", use_container_width=True)

if predict_button:
    if canvas_result.image_data is not None:
        img_data = canvas_result.image_data
        
        # The user draws in white on a black background. We can use any of the
        # RGB channels to get a grayscale representation of the drawing.
        # For example, the red channel (index 0).
        gray_channel = img_data[:, :, 0]

        # Convert to a PIL Image, which will be in 'L' (grayscale) mode
        pil_img = Image.fromarray(gray_channel)
        
        resized_img = pil_img.resize((image_size[1], image_size[0]), Image.LANCZOS)
        img_array = np.array(resized_img) / 255.0
        
        # Add channel dimension
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0)

        # Apply model-specific normalization
        if data_info["type"] == "bes":
            h_patches = data_info['h_patches']
            w_patches = data_info['w_patches']
            patch_size = 28 # MNIST digits are 28x28

            # Reshape into patches to apply normalization per-patch, like in training
            # Shape: (1, H, W) -> (num_patches, 1, 28, 28)
            patches = einops.rearrange(
                img_tensor, 'c (h ph) (w pw) -> (h w) c ph pw', 
                h=h_patches, w=w_patches, ph=patch_size, pw=patch_size
            )
            
            # Normalize each patch
            normalizer = torchvision.transforms.Normalize((0.1307,), (0.3081,))
            normalized_patches = normalizer(patches)
            
            # Rearrange back to a single image
            # Shape: (num_patches, 1, 28, 28) -> (1, H, W)
            img_tensor = einops.rearrange(
                normalized_patches, '(h w) c ph pw -> c (h ph) (w pw)',
                h=h_patches, w=w_patches
            )

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        device = select_device()
        img_tensor = img_tensor.to(device)
        
        with st.spinner("Predicting..."):
            predicted_tokens = predict_sequence(model, img_tensor)
            seq_str = "".join((str(int(x)) if x < 10 else "") for x in predicted_tokens)
            st.success(f"Predicted sequence: **{seq_str if seq_str else 'No digits detected'}**")

            c1, c2 = st.columns(2)
            with c1:
                st.write("Your Drawing (Processed):")
                st.image(pil_img, use_container_width=True)
            with c2:
                st.write("Resized for Model:")
                st.image(resized_img, use_container_width=True)
    else:
        st.warning("Please draw something on the canvas before predicting.")

if example_button:
    with st.spinner("Generating example and predicting..."):
        example_image_tensor = get_next_example_image(model_key, model, data_info)
        
        if example_image_tensor is not None:
            device = select_device()
            example_image_tensor = example_image_tensor.to(device)
            
            if example_image_tensor.dim() == 3:
                example_image_tensor = example_image_tensor.unsqueeze(0)

            predicted_tokens = predict_sequence(model, example_image_tensor)
            seq_str = "".join((str(int(x)) if x < 10 else "") for x in predicted_tokens)
            st.success(f"Predicted sequence: **{seq_str}**")

            st.image(example_image_tensor.cpu().squeeze().numpy(), caption="Example Image", use_container_width=True, clamp=True) 