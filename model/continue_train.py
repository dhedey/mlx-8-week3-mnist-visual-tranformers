# Run as uv run -m model.continue_train
import argparse
from .default_models import DEFAULT_MODEL_NAME, WANDB_PROJECT_NAME
from .trainer import ModelTrainerBase, TrainerOverrides
from .common import upload_model_artifact, ModelBase
import wandb
import os

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
    parser.add_argument(
        '--from-wandb', 
        type=str, default=None, 
        help='Continue from a wandb run ID, enabling W&B logging.'
    )
    parser.add_argument(
        '--wandb-project', 
        default=WANDB_PROJECT_NAME, 
        help='W&B project name (used if --from-wandb is set)'
    )
    parser.add_argument(
        '--no-upload-artifacts', 
        action='store_true', 
        help='Disable artifact uploading to W&B (if --from-wandb is set)'
        )
    args = parser.parse_args()

    overrides = TrainerOverrides(
        override_to_epoch=args.end_epoch,
        override_learning_rate=args.learning_rate,
        validate_after_epochs=args.validate_after_epochs,
    )

    if args.from_wandb:
        # W&B-enabled path
        run_id = args.from_wandb
        print(f"Loading model from wandb run: {run_id}")
        
        api = wandb.Api()
        
        artifact_name = f"model-{run_id}-best"
        project_path = f"{api.default_entity}/{args.wandb_project}"
        
        try:
            artifact = api.artifact(f"{project_path}/{artifact_name}:latest")
        except wandb.errors.CommError as e:
            print(f"Error: Could not find artifact '{artifact_name}' in project '{project_path}'. {e}")
            exit(1)

        download_path = artifact.download()
        pt_files = [f for f in os.listdir(download_path) if f.endswith('.pt')]
        if not pt_files:
            print(f"Error: No .pt file found in downloaded artifact at {download_path}")
            exit(1)
        model_file_path = os.path.join(download_path, pt_files[0])
        print(f"Found model file: {model_file_path}")

        trainer = ModelTrainerBase.load_with_model(
            model_name=args.model,
            model_path=model_file_path,
            overrides=overrides,
        )
        
        # Start a new W&B run for the continued training
        new_run_name = f"continued-from-{run_id}"
        config = {
            "source_run_id": run_id,
            "original_model_name": trainer.model.model_name,
            "model_config": trainer.model.config.to_dict(),
            "training_config": trainer.config.to_dict(),
            "overrides": vars(args)
        }
        trainer.model.model_name = new_run_name # Save new model with a new name
        wandb.init(project=args.wandb_project, name=new_run_name, config=config)

        try:
            if args.immediate_validation:
                print("Immediate validation enabled, running validation before training:")
                trainer.run_validation()

            results = trainer.train()

            if not args.no_upload_artifacts:
                print("Uploading artifacts...")
                model_name = trainer.model.model_name
                artifact_metadata = {
                    "source_run_id": run_id,
                    "model_config": trainer.model.config.to_dict(),
                    "training_config": trainer.config.to_dict(),
                    "final_validation_loss": results.last_validation.validation_loss,
                    "final_train_loss": results.last_training_epoch.average_loss,
                    "total_epochs": results.total_epochs,
                }
                
                model_path = ModelBase._model_path(model_name)
                best_model_path = ModelBase._model_path(f"{model_name}-best")

                if os.path.exists(model_path):
                    upload_model_artifact(
                        model_name=model_name, model_path=model_path,
                        artifact_name=f"model-{wandb.run.id}-final", metadata=artifact_metadata,
                        description=f"Final model from continued run {wandb.run.id}"
                    )

                if os.path.exists(best_model_path):
                    best_metadata = artifact_metadata.copy()
                    best_metadata["model_type"] = "best_validation"
                    upload_model_artifact(
                        model_name=f"{model_name}-best", model_path=best_model_path,
                        artifact_name=f"model-{wandb.run.id}-best", metadata=best_metadata,
                        description=f"Best validation model from continued run {wandb.run.id}"
                    )
        finally:
            wandb.finish()

    else:
        # Original, non-W&B path
        trainer = ModelTrainerBase.load_with_model(
            model_name=args.model,
            overrides=overrides,
        )

        if args.immediate_validation:
            print("Immediate validation enabled, running validation before training:")
            trainer.run_validation()

        trainer.train()



        
