# Run as uv run -m model.start_train

import argparse
from .common import select_device
from .models import DEFAULT_MODEL_PARAMETERS
from .trainer import EncoderOnlyModelTrainer

DEFAULT_MODEL_NAME = list(DEFAULT_MODEL_PARAMETERS.keys())[0]

if __name__ == "__main__":
    device = select_device()

    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL_NAME,
    )
    args = parser.parse_args()

    model_name = args.model

    print(f"Starting training model: {model_name}")

    parameters = DEFAULT_MODEL_PARAMETERS[model_name]
    model = parameters["model_class"](
        model_name=model_name,
        training_parameters=parameters["training"],
        model_parameters=parameters["model"],
    )

    trainer = EncoderOnlyModelTrainer(
        model=model.to(device),
        validate_after_epochs=1
    )
    trainer.train()



        
