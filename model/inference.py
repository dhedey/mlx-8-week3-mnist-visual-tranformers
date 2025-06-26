from .common import select_device, ModelBase
from .models import DigitSequenceModel, DEFAULT_MODEL_PARAMETERS
from .composite_dataset import BesCombine
import torch
import matplotlib.pyplot as plt
import numpy as np

def load_model(model_name, device):
    model = DigitSequenceModel(
        model_name=model_name,
        config=DEFAULT_MODEL_PARAMETERS["model"],
    )
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    model.to(device)
    model.eval()
    return model

def select_model(choice : str):
    if choice == "best2by2":
        model_name = "multi-digit-v1"
        model_path = ModelBase._model_path("best2by2")

    return {
        "model_name": model_name,
        "model_path": model_path,
    }

def predict_sequence(model, image) -> np.ndarray:
    #start with only the start token
    seq = torch.tensor([10] + [-1] * (model.config.max_sequence_length - 1))
    for i in range(model.config.max_sequence_length):
        logits = model(image, seq)[0]
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

if __name__ == "__main__":
    device = select_device()

    model_info = select_model("best2by2")
    model_name = model_info["model_name"]
    model_path = model_info["model_path"]

    model, state, config = ModelBase.load(model_name=model_name, device=device, model_path=model_path)

    #run the model on a test image, starting on the start token
    test_ds = BesCombine(train=False, h_patches=2, w_patches=2, length=100)
    test_image, _ = test_ds[0]
    out_seq = predict_sequence(model, test_image)
    #display the image and sequence of predicted tokens
    plt.imshow(test_image.squeeze(0).cpu().numpy(), cmap="gray")
    seq_str = "".join((str(int(x)) if x<10 else "<END>") for x in out_seq)
    plt.title(f"Predicted sequence: {seq_str}")
    plt.show()