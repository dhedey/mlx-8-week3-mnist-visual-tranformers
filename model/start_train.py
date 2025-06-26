# Run as uv run -m model.start_train

import argparse
from .common import select_device, ModelBase, upload_model_artifact
from .models import DEFAULT_MODEL_PARAMETERS
from .trainer import EncoderOnlyModelTrainer, TrainerOverrides, DigitSequenceModelTrainer
import wandb
import os

DEFAULT_MODEL_NAME = list(DEFAULT_MODEL_PARAMETERS.keys())[0]
PROJECT_NAME = "week3-mnist-transformers"

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
    parser.add_argument(
        '--wandb', 
        action='store_true', 
        help='Enable wandb logging'
    )
    parser.add_argument(
        '--wandb-project', 
        default=PROJECT_NAME, 
        help=f'W&B project name (default: {PROJECT_NAME})'
    )
    parser.add_argument(
        '--no-upload-artifacts',
        action='store_true',
        help='Disable artifact uploading to W&B (if wandb is enabled)'
    )
    
    args = parser.parse_args()

    model_name = args.model

    parameters = DEFAULT_MODEL_PARAMETERS[model_name]

    if args.wandb:
        config = {
            "model_name": model_name,
            "model_config": parameters["model"].to_dict(),
            "training_config": parameters["training"].to_dict(),
        }
        wandb.init(project=args.wandb_project, config=config)

    model = parameters["model_class"](
        model_name=model_name,
        config=parameters["model"],
    ).to(device)

    trainer = parameters["model_trainer"](
        model=model,
        config=parameters["training"],
    )
    results = trainer.train()

    if args.wandb and not args.no_upload_artifacts:
        print("Uploading artifacts...")
        artifact_metadata = {
            "model_name": model_name,
            "model_config": parameters["model"].to_dict(),
            "training_config": parameters["training"].to_dict(),
            "final_validation_loss": results.last_validation.validation_loss,
            "final_train_loss": results.last_epoch.average_loss,
            "total_epochs": results.total_epochs,
        }
        
        model_path = ModelBase._model_path(model_name)
        best_model_path = ModelBase._model_path(f"{model_name}-best")

        if os.path.exists(model_path):
            upload_model_artifact(
                model_name=model_name,
                model_path=model_path,
                artifact_name=f"{model_name}-final",
                metadata=artifact_metadata,
                description=f"Final model: {model_name}"
            )

        if os.path.exists(best_model_path):
            best_metadata = artifact_metadata.copy()
            best_metadata["model_type"] = "best_validation"
            upload_model_artifact(
                model_name=f"{model_name}-best",
                model_path=best_model_path,
                artifact_name=f"{model_name}-best",
                metadata=best_metadata,
                description=f"Best validation model: {model_name}"
            )

    if args.wandb:
        wandb.finish()



        
