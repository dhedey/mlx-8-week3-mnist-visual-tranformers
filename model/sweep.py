#!/usr/bin/env python3
# Run as uv run -m model.sweep
"""
Weights & Biases Hyperparameter Sweep Script

This script programmatically creates and runs wandb sweeps for the dual encoder search model.
It provides more control over the sweep process compared to the CLI-based approach.
"""

import wandb
import os

from .models import TrainingConfig, SingleDigitModel, SingleDigitModelConfig
from .trainer import SingleDigitModelTrainer
from .common import select_device

from .default_models import WANDB_PROJECT_NAME

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
        'batch_size': {
            'values': [128, 256]
        },
        'embedding_size': {
            'values': [8, 16, 32]
        },
        'learning_rate': {
            'min': 0.001,
            'max': 0.03,
            'distribution': 'log_uniform_values'
        },
        'kq_size': {
            'values': [8, 16, 32]
        },
        'v_size': {
            'values': [8, 16, 32]
        },
        'positional_embedding': {
            'values': ["learned-bias", "none"]
        },
        'epochs': {
            'values': [20]
        },
    }
}

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

        match config.positional_embedding:
            case "learned-bias":
                add_positional_bias = True
            case "none":
                add_positional_bias = False
            case _:
                raise ValueError(f"Unknown positional embedding type: {config.positional_embedding}")

        training_parameters = TrainingConfig(
            batch_size=config.batch_size,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            early_stopping=enable_early_stopping,
            early_stopping_patience=early_stopping_patience,
        )

        model_parameters = SingleDigitModelConfig(
            encoder_blocks=config.encoder_blocks,
            embedding_size=config.embedding_size,
            kq_dimension=config.kq_size,
            v_dimension=config.v_size,
            mlp_hidden_dimension=4 * config.embedding_size,  # Typical in transformers
            add_positional_bias=add_positional_bias,
        )

        model = SingleDigitModel(
            model_name="sweep-run",
            training_parameters=training_parameters,
            model_parameters=model_parameters,
        )

        trainer = SingleDigitModelTrainer(model=model.to(device))
        results = trainer.train()
        
        # Log final metrics
        log_data = {
            "final_train_average_loss": results.last_training_epoch.average_loss,
            "best_train_average_loss": results.best_training_epoch.average_loss,
            "final_validation_loss": results.last_validation.validation_loss,
            "best_validation_loss": results.best_validation.validation_loss,
            "total_epochs": results.total_epochs,
        }
        wandb.log(log_data)
        
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
    parser.add_argument('--project', default=WANDB_PROJECT_NAME,
                        help=f'W&B project name (default: {WANDB_PROJECT_NAME})')
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

    # Run sweep
    if args.sweep_id:
        run_existing_sweep(args.sweep_id, args.project, args.count)
    else:
        sweep_id = create_and_run_sweep(config, args.project, args.count)
        print(f"\n💾 Save this sweep ID for future use: {sweep_id}")


if __name__ == '__main__':
    main()
