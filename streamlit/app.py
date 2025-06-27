import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import numpy as np
import torch
import random
import io
import sys
import os
import torchvision
import torchvision.transforms.v2 as v2
import einops
import time
from uuid import uuid4
import streamlit.components.v1 as components

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.inference import get_data_info, select_model, predict_sequence
from model.models import DigitSequenceModel
from model.common import select_device
from model.composite_dataset import BesCombine, CompositeDataset, DavidCompositeDataset


st.set_page_config(page_title="Overlord's Vision", page_icon="👀")

st.title("Overlord's Vision")
st.write("A multi-digit MNIST sequence predictor.")

# Invert the display_names dictionary to map display names back to keys
display_names = {
    "scrambled-v2": "Shuffletastic",
    "best4by4_var": "4 x 4 on-grid",
    "best2by2": "2 x 2 on-grid (fixed length)",
}
name_to_key = {v: k for k, v in display_names.items()}

# --- Model Selection ---
mode_col, model_select_col = st.columns([5, 5])

with mode_col:
    mode = st.selectbox(
        "Choose a mode:",
        options=["Generate", "Draw", "Game"],
        index=0,
    )
with model_select_col:
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
    torch.manual_seed(random.randint(1, 100000))

    if data_type == "bes":
        h_patches = _data_info["h_patches"]
        w_patches = _data_info["w_patches"]
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
        
        david_generator = DavidCompositeDataset(
            image_dataset=mnist_ds,
            output_width=_model.config.encoder.image_width,
            output_height=_model.config.encoder.image_height,
            length=10,
            line_height_min=16,
            line_height_max=64,
        )
        return iter(david_generator)
    elif data_type == "david-v2":
        data_folder = os.path.join(os.path.dirname(__file__), "..", "model", "datasets")
        
        david_generator = DavidCompositeDataset(
            train=False,
            length=10,
            output_width=_model.config.encoder.image_width,
            output_height=_model.config.encoder.image_height,
            line_height_min=16,
            line_height_max=30,
            line_spacing_min=2,
            line_spacing_max=20,
            horizontal_padding_min=2,
            horizontal_padding_max=60,
            left_margin_offset=0,
            first_line_offset=0,
            image_scaling_min=0.8,
            max_sequence_length=10,
        )
        return iter(david_generator)
    else:
        st.error(f"Example generation not supported for type: {data_type}")
        return None

def get_next_example_image(model_key, model, data_info) -> torch.Tensor:
    """Gets the next image from a cached generator, creating one if needed."""
    if 'image_generator' not in st.session_state or st.session_state.get('model_key') != model_key:
        st.session_state.image_generator = create_image_generator(model, data_info)
        st.session_state.model_key = model_key

    generator = st.session_state.image_generator
        
    try:
        data_type = data_info["type"]
        if data_type == "bes":
            example_image_tensor, labels = next(generator)
        elif data_type == "david" or data_type == "david-v2":
            img_batch, labels, _ = next(generator)
            example_image_tensor = img_batch[0]
        else:
            raise ValueError(f"Unsupported data type for example generation: {data_type}")
        
        labels = [label.item() for label in labels if label.item() >= 0 and label.item() <= 9]
        return example_image_tensor, labels
    except StopIteration:
        # st.warning("Image generator exhausted. Re-initializing...")
        # Clear the old generator and recursively call to get a new one
        del st.session_state['image_generator']
        return get_next_example_image(model_key, model, data_info)

with st.spinner(f"Loading model: {model_display_name}..."):
    model = load_model(model_key)

data_info = get_data_info(model_key)
image_size = data_info["image_size"]


canvas_height = 400
canvas_width = 400

if mode == "Draw":

    # --- Drawable Canvas ---
    st.write(f"Draw a digit sequence. The model expects an image of size {image_size[1]}x{image_size[0]}. Your drawing will be resized.")

    # Adjust stroke width based on the selected model
    if model_key == "best4by4_var":
        stroke_width = 5
    else:
        stroke_width = 10

    main_col, prediction_col = st.columns([6, 4])

    with main_col:
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="freedraw",
            key="canvas",
        )
        if data_info["type"] == "bes":
            # JavaScript overlay grid aligned precisely on the canvas
            cell_h = canvas_height // data_info["h_patches"]
            cell_w = canvas_width  // data_info["w_patches"]
            grid_uid = f"grid-{model_key}" # Unique ID for the grid

            components.html(f"""
                <script>
                    function refreshGrid() {{
                        // Remove all existing grid overlays
                        window.parent.document.querySelectorAll('[id^="grid-"]').forEach(el => el.remove());

                        const wrapper = window.parent.document.querySelector('div.element-container iframe')?.parentElement;
                        if (!wrapper) return;

                        wrapper.style.position = 'relative';
                        const grid = document.createElement('div');
                        grid.id = '{grid_uid}';
                        grid.style.position = 'absolute';
                        grid.style.top = '0';
                        grid.style.left = '0';
                        grid.style.width = '{canvas_width}px';
                        grid.style.height = '{canvas_height}px';
                        grid.style.pointerEvents = 'none';
                        grid.style.backgroundImage =
                        'linear-gradient(to right, rgba(255,255,255,0.4) 2px, transparent 2px),' +
                        'linear-gradient(to bottom, rgba(255,255,255,0.4) 2px, transparent 2px)';
                        grid.style.backgroundSize = '{cell_w}px {cell_h}px';
                        wrapper.appendChild(grid);
                    }}
                    // Use a timeout to run after the DOM has updated
                    setTimeout(refreshGrid, 50);
                </script>
                """,
                height=0
            )
        else:
            # If model is not grid-based, ensure no grids are shown
            components.html("""
                <script>
                    function cleanupGrid() {{
                        window.parent.document.querySelectorAll('[id^="grid-"]').forEach(el => el.remove());
                    }}
                    setTimeout(cleanupGrid, 50);
                </script>
                """,
                height=0
            )


    # --- Prediction Logic ---
    with prediction_col:
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
                # st.write("Resized for Model:")
                # st.image(resized_img, use_container_width=True)
                predicted_tokens = predict_sequence(model, img_tensor)
                seq_str = "".join((str(int(x)) if x < 10 else "") for x in predicted_tokens)
                st.warning(f"Predicted sequence:\n\n**{seq_str if seq_str else 'No digits detected'}**")
                    
        else:
            st.warning("Please draw something on the canvas before predicting.")

elif mode == "Generate":
    main_col, prediction_col = st.columns([6, 4])
    with st.spinner("Generating example and predicting..."):
        with main_col:
            example_image_tensor, actual_labels = get_next_example_image(model_key, model, data_info)
            
            device = select_device()
            example_image_tensor = example_image_tensor.to(device)
            
            if example_image_tensor.dim() == 3:
                example_image_tensor = example_image_tensor.unsqueeze(0)

            st.image(example_image_tensor.cpu().squeeze().numpy(), caption="Example Image", use_container_width=True, clamp=True)
                
        with prediction_col:
            if example_image_tensor is not None:

                actual_seq_str = "".join((str(int(x)) if x < 10 else "") for x in actual_labels)

                predicted_tokens = predict_sequence(model, example_image_tensor)

                seq_str = "".join((str(int(x)) if x < 10 else "") for x in predicted_tokens)
                text = f"Predicted sequence:\n\n**{seq_str if seq_str else 'No digits detected'}**\n\nActual sequence:\n\n**{actual_seq_str}**"
                if seq_str == actual_seq_str:
                    st.success(text)
                else:
                    st.error(text)

            st.button("Regenerate...")

elif mode == "Game":
    if 'user_correct' not in st.session_state:
        st.session_state['user_correct'] = 0
    if 'ai_correct' not in st.session_state:
        st.session_state['ai_correct'] = 0
    if 'attempts' not in st.session_state:
        st.session_state['attempts'] = 0

    user_correct = st.session_state.get('user_correct', 0)
    ai_correct = st.session_state.get('ai_correct', 0)
    attempts = st.session_state.get('attempts', 0)
    round_state = st.session_state.get('round_state', "start")

    st.write(f"**User Correct:** {user_correct} | **AI Correct:** {ai_correct} | **Attempts:** {attempts}")

    next_round = st.button("Next Round", key="next_round")

    if next_round:
        match round_state:
            case "start":
                round_state = "see"
            case "see":
                round_state = "guess"
            case "guess":
                round_state = "round_result"
            case "round_result":
                round_state = "start"
        st.session_state['round_state'] = round_state

    match round_state:
        case "start":
            example_image_tensor, actual_labels = get_next_example_image(model_key, model, data_info)
            actual_labels = "".join((str(int(x)) if x < 10 else "") for x in actual_labels)
            if example_image_tensor.dim() == 3:
                example_image_tensor = example_image_tensor.unsqueeze(0)

            start_time = time.time()
            predicted_tokens = predict_sequence(model, example_image_tensor)
            ai_labels = "".join((str(int(x)) if x < 10 else "") for x in predicted_tokens)
            prediction_time = time.time() - start_time
            
            st.session_state['image'] = example_image_tensor
            st.session_state['ai_labels'] = ai_labels
            st.session_state['actual_labels'] = actual_labels
            st.session_state['prediction_time'] = prediction_time
            st.session_state['user_time_ms'] = prediction_time * 1000

            st.write(f"The AI predicted this round in {prediction_time * 1000:0.0f} ms, you will be given the same! Good luck.")
            st.write("Press 'Next Round' to start the next round.")
        case "see":
            col, _ = st.columns([5, 5])
            with col:
                st.image(
                    st.session_state['image'].cpu().squeeze().numpy(),
                    caption="Example Image",
                    use_container_width=True,
                    clamp=True,
                )

                components.html("""
                    <script>
                        function complete_game() {{
                            window.parent.document.querySelectorAll('.st-key-next_round button').forEach(el => el.click());
                        }}
                        setTimeout(complete_game, """ + str(st.session_state['user_time_ms']) + """);
                    </script>
                    """,
                    height=0
                )
        case "guess":
            data = st.text_input(
                "Enter your guess (sequence of digits, e.g. '1234'):",
                key="user_guess",
                placeholder="Type your guess here...",
            )
            st.session_state['user_labels'] = data
            st.write("Press 'Next Round' to submit your guess.")
        case "round_result":
            st.write("Round Results:")
            user_correct = st.session_state['user_labels'] == st.session_state['actual_labels']
            ai_correct = st.session_state['ai_labels'] == st.session_state['actual_labels']
            st.warning(f"**Actual Sequence:** {st.session_state['actual_labels']}")

            st.session_state['attempts'] += 1
            if user_correct:
                st.success(f"**Your Guess:** {st.session_state['user_labels']}")
                st.session_state['user_correct'] += 1
            else:
                st.error(f"**Your Guess:** {st.session_state['user_labels']}")

            if ai_correct:
                st.success(f"**AI's Guess:** {st.session_state['ai_labels']}")
                st.session_state['ai_correct'] += 1
            else:
                st.error(f"**AI's Guess:** {st.session_state['ai_labels']}")

            st.write("Press 'Next Round' to continue.")

        

