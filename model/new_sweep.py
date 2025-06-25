#!/usr/bin/env python3
# Run as uv run -m model.sweep
"""
Weights & Biases Hyperparameter Sweep Script

This script programmatically creates and runs wandb sweeps for the dual encoder search model.
It provides more control over the sweep process compared to the CLI-based approach.
"""

import wandb
import os

from .models import DigitSequenceModel, DigitSequenceModelConfig, ImageEncoderConfig, EncoderBlockConfig, DecoderBlockConfig
from .trainer import DigitSequenceModelTrainer
from .common import select_device, TrainingConfig, TrainerOverrides

PROJECT_NAME = "week3-mnist-transformers"

# Sweep configuration - equivalent to wandb_sweep.yaml but in Python
# https://docs.wandb.ai/guides/sweeps/sweep-config-keys/
SWEEP_CONFIG = {
    'method': 'bayes',  # Can be 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'final_validation_average_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'encoder_blocks': {
            'values': [1, 3, 5],
        },
        'decoder_blocks': {
            'values': [1, 2, 3],
        },
        'batch_size': {
            'values': [256, 512, 1024]
        },
        'embedding_size': {
            'values': [32, 64, 128]
        },
        'learning_rate': {
            'min': 0.0001,
            'max': 0.001,
            'distribution': 'log_uniform_values'
        },
        'num_heads': {
            'values': [1, 2, 4, 8]
        },
        'kq_size': {
            'values': [16, 32, 64]
        },
        'v_size': {
            'values': [16, 32, 64]
        },
        'positional_embedding': {
            'values': ["learned-bias"]
        },
        'epochs': {
            'values': [10]
        },
    }
}

def upload_model_artifact(model_name: str, model_path: str, artifact_name: str = None, 
                         metadata: dict = None, description: str = None):
    """
    Upload a model as a wandb artifact.
    
    Args:
        model_name: Name of the model (used for artifact naming if artifact_name not provided)
        model_path: Path to the saved model file
        artifact_name: Optional custom artifact name (defaults to model_name)
        metadata: Optional metadata dictionary to include with the artifact
        description: Optional description for the artifact
    """
    if not os.path.exists(model_path):
        print(f"⚠️  Model file not found: {model_path}")
        return None
    
    # Use model_name as artifact name if not provided
    if artifact_name is None:
        artifact_name = model_name
    
    # Create artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=description or f"Trained model: {model_name}",
        metadata=metadata or {}
    )
    
    # Add the model file
    artifact.add_file(model_path, name=f"{model_name}.pt")
    
    # Log the artifact
    wandb.log_artifact(artifact)
    print(f"📦 Uploaded model artifact: {artifact_name}")
    
    return artifact

def train_sweep_run():
    """
    Single training run for wandb sweep.
    This function is called by the sweep agent for each hyperparameter combination.
    """
    # Initialize wandb run
    wandb.init()
    
    try:
        config = wandb.config
        
        print(f"\n🚀 Starting sweep run")
        device = select_device()

        # Check for early stopping from environment
        enable_early_stopping = os.environ.get('SWEEP_EARLY_STOPPING', 'false').lower() == 'true'
        early_stopping_patience = int(os.environ.get('SWEEP_EARLY_STOPPING_PATIENCE', '5'))
        
        # Check if artifact upload is enabled
        upload_artifacts = os.environ.get('SWEEP_UPLOAD_ARTIFACTS', 'false').lower() == 'true'

        match config.positional_embedding:
            case "learned-bias":
                add_positional_bias = True
            case "none":
                add_positional_bias = False
            case _:
                raise ValueError(f"Unknown positional embedding type: {config.positional_embedding}")

        training_config = TrainingConfig(
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            early_stopping=enable_early_stopping,
            early_stopping_patience=early_stopping_patience,
        )

        model_config = DigitSequenceModelConfig(
            max_sequence_length=11,
            encoder=ImageEncoderConfig(
                image_width=56,
                image_height=56,
                image_patch_width=7,
                image_patch_height=7,
                embedding_dimension=config.embedding_size,
                encoder_block_count=config.encoder_blocks,
                encoder_block=EncoderBlockConfig(
                    kq_dimension=config.kq_size,
                    v_dimension=config.v_size,
                    embedding_dimension=config.embedding_size,
                    num_heads=config.num_heads,
                    mlp_hidden_dimension=4 * config.embedding_size,
                    mlp_dropout=0.2,
                ),
            ),
            decoder_block_count=config.decoder_blocks,
            decoder_block=DecoderBlockConfig(
                encoder_embedding_dimension=config.embedding_size,
                decoder_embedding_dimension=config.embedding_size,

                self_attention_kq_dimension=config.kq_size,
                self_attention_v_dimension=config.v_size,
                self_attention_heads=config.num_heads,

                cross_attention_kq_dimension=config.kq_size,
                cross_attention_v_dimension=config.v_size,
                cross_attention_heads=config.num_heads,

                mlp_hidden_dimension=4 * config.embedding_size,
                mlp_dropout=0.2,
            ),
        )

        # Use wandb run id for unique model naming
        run_id = wandb.run.id
        model_name = f"sweep-run-{run_id}"

        model = DigitSequenceModel(
            model_name=model_name,
            config=model_config,
        )

        trainer = DigitSequenceModelTrainer(model=model.to(device), config=training_config, overrides=TrainerOverrides())
        results = trainer.train()
        
        # Log final metrics
        log_data = {
            "final_train_average_loss": results.last_epoch.average_loss,
            "total_epochs": results.total_epochs,
        }
        for key, value in results.last_validation.model_dump().items():
            #note: Keys are already prefixed with "validation_"
            log_data[f"final_{key}"] = value
        wandb.log(log_data)
        
        # Upload model artifacts if enabled
        if upload_artifacts:
            try:
                # Prepare metadata for the artifact
                artifact_metadata = {
                    "config": dict(config),
                    "final_validation_loss": results.last_validation.validation_loss,
                    "final_train_loss": results.last_epoch.average_loss,
                    "total_epochs": results.total_epochs,
                    "wandb_run_id": run_id,
                    "model_architecture": "ImageSequenceTransformer",
                }
                
                # Get model save paths
                from .common import PersistableModel
                model_path = PersistableModel._model_path(model_name)
                best_model_path = PersistableModel._model_path(f"{model_name}-best")
                
                # Upload final model
                if os.path.exists(model_path):
                    upload_model_artifact(
                        model_name=model_name,
                        model_path=model_path,
                        artifact_name=f"model-{run_id}",
                        metadata=artifact_metadata,
                        description=f"Final model from sweep run {run_id}"
                    )
                
                # Upload best model if it exists
                if os.path.exists(best_model_path):
                    best_metadata = artifact_metadata.copy()
                    best_metadata["model_type"] = "best_validation"
                    upload_model_artifact(
                        model_name=f"{model_name}-best",
                        model_path=best_model_path,
                        artifact_name=f"model-{run_id}-best",
                        metadata=best_metadata,
                        description=f"Best validation model from sweep run {run_id}"
                    )
                    
            except Exception as e:
                print(f"⚠️  Failed to upload artifacts: {e}")
                # Don't fail the entire run if artifact upload fails
        
        print(f"✅ Sweep run completed!")
        
    except Exception as e:
        print(f"❌ Sweep run failed: {e}")
        # Log the failure
        wandb.log({"status": "failed", "error": str(e)})
        raise
    
    finally:
        # Ensure wandb run is properly finished
        wandb.finish()


def create_and_run_sweep(config, project_name, count=10):
    """
    Create and run a wandb sweep programmatically.
    
    Args:
        config: Sweep configuration dictionary (defaults to SWEEP_CONFIG)
        project_name: W&B project name
        count: Number of runs to execute in the sweep
    """
    print(f"🔧 Creating sweep with {config['method']} optimization...")
    print(f"📊 Target metric: {config['metric']['name']} ({config['metric']['goal']})")
    
    # Create the sweep
    sweep_id = wandb.sweep(config, project=project_name)
    print(f"✅ Sweep created with ID: {sweep_id}")
    print(f"🌐 View sweep at: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")
    
    # Run the sweep
    print(f"🏃 Starting sweep agent with {count} runs...")
    wandb.agent(sweep_id, train_sweep_run, project=project_name, count=count)
    
    print(f"🎉 Sweep completed!")
    return sweep_id


def run_existing_sweep(sweep_id, project_name, count=10):
    """
    Run an existing sweep by ID.
    
    Args:
        sweep_id: The ID of an existing sweep
        count: Number of additional runs to execute
    """
    print(f"🔄 Joining existing sweep: {sweep_id} against {project_name}")
    wandb.agent(sweep_id, train_sweep_run, project=project_name, count=count)


def main():
    """
    Main function with different sweep options.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter sweeps')
    parser.add_argument('--project', default=PROJECT_NAME,
                        help=f'W&B project name (default: {PROJECT_NAME})')
    parser.add_argument('--count', type=int, default=20,
                        help='Number of sweep runs (default: 20)')
    parser.add_argument('--sweep-id', type=str,
                        help='Join existing sweep by ID instead of creating new one')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just show the configuration without running')
    parser.add_argument('--early-stopping', action='store_true',
                        help='Enable early stopping for all sweep runs')
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                        help='Number of epochs to wait before early stopping (default: 5)')
    parser.add_argument('--upload-artifacts', action='store_true',
                        help='Upload trained models as wandb artifacts')
    
    args = parser.parse_args()
    
    # Select configuration
    config = SWEEP_CONFIG
    print("📋 Using default Bayesian optimization configuration")
    
    if args.dry_run:
        print("\n🔍 Sweep configuration:")
        import json
        print(json.dumps(config, indent=2))
        return
    
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Enable early stopping if requested
    if args.early_stopping:
        os.environ["SWEEP_EARLY_STOPPING"] = "True"
        os.environ["SWEEP_EARLY_STOPPING_PATIENCE"] = str(args.early_stopping_patience)
    
    # Enable artifact upload if requested
    if args.upload_artifacts:
        os.environ["SWEEP_UPLOAD_ARTIFACTS"] = "True"
        print("📦 Artifact upload enabled - models will be saved to wandb")

    # Run sweep
    if args.sweep_id:
        run_existing_sweep(args.sweep_id, args.project, args.count)
    else:
        sweep_id = create_and_run_sweep(config, args.project, args.count)
        print(f"\n💾 Save this sweep ID for future use: {sweep_id}")


if __name__ == '__main__':
    main()
