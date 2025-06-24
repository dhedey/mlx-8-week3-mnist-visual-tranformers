# Run as uv run -m model.start_train

import argparse
from .common import select_device
from .models import DEFAULT_MODEL_PARAMETERS
from .trainer import EncoderOnlyModelTrainer, TrainerParameters

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
    training_params = parameters["training"]
    if args.early_stopping:
        training_params.early_stopping = True
        print("Early stopping enabled via command line argument")

    model = parameters["model_class"](
        model_name=model_name,
        training_parameters=training_params,
        model_parameters=parameters["model"],
    ).to(device)

    trainer = EncoderOnlyModelTrainer(
        model=model,
        parameters=TrainerParameters(),
    )
    trainer.train()



        
