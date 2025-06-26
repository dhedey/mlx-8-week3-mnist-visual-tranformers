from .common import select_device, ModelBase, ModuleConfig
from .models import DigitSequenceModel
from .default_models import DEFAULT_MODEL_PARAMETERS
from .composite_dataset import BesCombine
import torch
import matplotlib.pyplot as plt
import numpy as np
from pydantic import ValidationError
from typing import Self

# def legacy_load(model_name: str, model_path:str, device: str) -> tuple[Self, ModuleConfig]:
#     loaded_model_data = torch.load(model_path, map_location=device)
#     print(f"Model data (legacy) read from {model_path}")
#     loaded_class_name = loaded_model_data["model"]["class_name"]

#     registered_types = ModelBase.registered_types
#     if loaded_class_name not in registered_types:
#         raise ValueError(f"Model class {loaded_class_name} is not a known Model. Available classes: {list(registered_types.keys())}")
#     model_class = registered_types[loaded_class_name]

#     model_name_loaded = loaded_model_data["model"]["model_name"]
#     model_weights = loaded_model_data["model"]["weights"]
#     model_config = model_class.config_class.from_dict(loaded_model_data["model"]["config"])

#     model = model_class(
#         model_name=model_name_loaded,
#         config=model_config,
#     )
#     model.load_state_dict(model_weights)
#     model.to(device)
#     model.eval()
#     return model, model_config


def select_model(choice : str):
    match choice:
        case "best2by2":
            model_name = "multi-digit-v1"
            model_path = ModelBase.model_path("best2by2")
        case "best4by4_var":
            model_name = "multi-digit-v1"
            model_path = ModelBase.model_path("best4by4")
        case _:
            raise ValueError(f"Invalid model choice: {choice}")

    return {
        "model_name": model_name,
        "model_path": model_path,
    }

def get_data_info(choice : str):
    match choice:
        case "best2by2":
            return {
                "h_patches": 2,
                "w_patches": 2,
                "image_size": (56, 56),
                "variable_length": False,
            }
        case "best4by4_var":
            return {
                "h_patches": 4,
                "w_patches": 4,
                "image_size": (112, 112),
                "variable_length": True,
            }
        case _:
            raise ValueError(f"Invalid model choice: {choice}")

def predict_sequence(model, image) -> np.ndarray:
    #start with only the start token
    seq = torch.tensor([10] + [-1] * (model.config.max_sequence_length))
    for i in range(model.config.max_sequence_length):
        logits = model(image, seq[:-1])[0]
        #get the logits for the next token
        next_token_logits = logits[i, :]
        #sample the next token
        next_token = torch.argmax(next_token_logits)
        #add the next token to the sequence
        seq[i + 1] = next_token
        #stop if the next token is the end token
        if next_token == 10:
            break
    return seq[seq != -1][1:].cpu().numpy()

def display_test_images(model, data_info):
    h_patches = data_info["h_patches"]
    w_patches = data_info["w_patches"]
    p_skip = 0.2 if data_info["variable_length"] else 0
    test_ds = BesCombine(train=False, h_patches=h_patches, w_patches=w_patches, length=100, p_skip=p_skip)
    for test_image, _ in test_ds:
        out_seq = predict_sequence(model, test_image)
        seq_str = "".join((str(int(x)) if x<10 else "<END>") for x in out_seq)
        plt.imshow(test_image.squeeze(0).cpu().numpy(), cmap="gray")
        plt.title(f"Predicted sequence: {seq_str}")
        plt.show()
        yield test_image, out_seq

if __name__ == "__main__":
    device = select_device()

    #load the model and data info   
    name = "best4by4_var"
    all_names = ["best4by4_var", "best2by2"]
    model_info = select_model(name)
    data_info = get_data_info(name)

    model_name = model_info["model_name"]
    model_path = model_info["model_path"]

    model = DigitSequenceModel.load_for_evaluation(model_path=model_path, device=device)

    for test_image, out_seq in display_test_images(model, data_info):
        user_input = input("Displaying test predictions. Press Enter to continue, 'q' to quit...")
        if user_input == "q":
            break