# Run as uv run -m model.continue_train
import argparse
from .models import ModelBase, DEFAULT_MODEL_PARAMETERS
from .trainer import ModelTrainer

DEFAULT_MODEL_NAME = DEFAULT_MODEL_PARAMETERS.keys()[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continue training a model')
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        '--end-epoch',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--immediate-validation',
        type=bool,
        default=False,
    )
    args = parser.parse_args()

    model, training_state = ModelBase.load_for_training(
        model_name=args.model
    )

    trainer = ModelTrainer(
        model=model,
        continuation=training_state,
        override_to_epoch=args.end_epoch,
        immediate_validation=args.immediate_validation,
    )
    trainer.train()



        
