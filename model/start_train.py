# Run as uv run -m model.start_train

import argparse
from .common import select_device
from .models import DEFAULT_MODEL_PARAMETERS
from .trainer import EncoderOnlyModelTrainer, TrainerOverrides

DEFAULT_MODEL_NAME = list(DEFAULT_MODEL_PARAMETERS.keys())[0]

if __name__ == "__main__":
    device = select_device()

    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        '--early-stopping',
        action='store_true',
        help='Enable early stopping during training'
    )
    args = parser.parse_args()

    model_name = args.model

    print(f"Starting training model: {model_name}")

    parameters = DEFAULT_MODEL_PARAMETERS[model_name]

    model = parameters["model_class"](
        model_name=model_name,
        config=parameters["model"],
    ).to(device)

    trainer = parameters["model_trainer"](
        model=model,
        config=parameters["training"],
    )
    trainer.train()



        
