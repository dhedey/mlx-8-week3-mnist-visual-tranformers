# Run as uv run -m model.continue_train
import argparse
from .models import ModelBase, DEFAULT_MODEL_PARAMETERS
from .trainer import ModelTrainerBase, TrainerOverrides

DEFAULT_MODEL_NAME = list(DEFAULT_MODEL_PARAMETERS.keys())[0]

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
        '--learning-rate',
        type=float,
        default=None,
    )
    parser.add_argument(
        '--immediate-validation',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--validate-after-epochs',
        type=int,
        default=1,
    )
    args = parser.parse_args()

    trainer = ModelTrainerBase.load_with_model(
        model_name=args.model,
        overrides=TrainerOverrides(
            override_to_epoch=args.end_epoch,
            override_learning_rate=args.learning_rate,
            validate_after_epochs=args.validate_after_epochs,
        ),
    )

    if args.immediate_validation:
        print("Immediate validation enabled, running validation before training:")
        trainer.run_validation()

    trainer.train()



        
