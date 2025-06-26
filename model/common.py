from dataclasses import dataclass
import torch.optim as optim
import torch.nn as nn
import torch
import os
import time
import wandb
import pathlib
import inspect
from typing import Optional, Self
from pydantic import BaseModel as PydanticBaseModel, Field

class PersistableData(PydanticBaseModel):
    def to_dict(self):
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)

class ModuleConfig(PersistableData):
    pass

def select_device():
    DEVICE_IF_MPS_SUPPORT = 'cpu' # or 'mps' - but it doesn't work well with EmbeddingBag
    device = torch.device('cuda' if torch.cuda.is_available() else DEVICE_IF_MPS_SUPPORT if torch.backends.mps.is_available() else 'cpu')
    
    print(f'Selected device: {device}')
    return device

class TrainingConfig(PersistableData):
    batch_size: int
    epochs: int
    learning_rate: float
    warmup_epochs: int = 0 # Number of epochs to warm up the learning rate
    optimizer: str = "adam" # "adam" or "adamw". Default should be changed to "adamw" in the future.    early_stopping: bool = False
    early_stopping: bool = False
    early_stopping_patience: int = 5

class ValidationResults(PersistableData, extra="allow"):
    epoch: int
    # This should be some measure of how well the validation went. Low is good.
    # It could simply be the average loss over the validation set, or some other better metric.
    validation_loss: float
    # You can add more fields here simply by adding them to the constructor

@dataclass
class BatchResults:
    total_loss: torch.Tensor # Singleton float tensor
    num_samples: int
    intermediates: dict

class EpochTrainingResults(PersistableData):
    epoch: int
    average_loss: float
    num_samples: int

class FullTrainingResults(PersistableData):
    total_epochs: int
    last_validation: ValidationResults
    last_epoch: EpochTrainingResults

class TrainingState(PersistableData):
    epoch: int
    optimizer_state: dict
    model_trainer_class_name: str
    total_training_time_seconds: float
    latest_training_results: EpochTrainingResults
    latest_validation_results: ValidationResults

class TrainerOverrides(PydanticBaseModel):
    override_to_epoch: Optional[int] = None
    override_learning_rate: Optional[float] = None
    validate_after_epochs: int = 1

class ModelBase(nn.Module):
    registered_types: dict[str, type] = {}
    config_class: type[ModuleConfig] = ModuleConfig

    def __init__(self, model_name: str, config: ModuleConfig):
        super().__init__()
        self.model_name = model_name
        self.config=config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Register the class name in the ModelBase.registered_types dictionary
        if cls.__name__ in ModelBase.registered_types:
            raise ValueError(f"Model {cls.__name__} is a duplicate classname. Use a new class name.")
        ModelBase.registered_types[cls.__name__] = cls

        # Register the config class in cls.config_class
        init_signature = inspect.signature(cls.__init__)
        init_params = init_signature.parameters

        if "config" not in init_params:
            raise ValueError(f"Model {cls.__name__} must have a 'config' parameter in its __init__ method.")

        config_param_class = init_params["config"].annotation

        if not issubclass(config_param_class, ModuleConfig):
            raise ValueError(f"Model {cls.__name__} has a 'config' parameter in its __init__ method called {config_param_class}, but this class does not derive from ModuleConfig.")

        cls.config_class = config_param_class

    def get_device(self):
        return next(self.parameters()).device
    
    @staticmethod
    def _model_path(model_name: str) -> str:
        model_folder = os.path.join(os.path.dirname(__file__), "saved")
        return os.path.join(model_folder, f"{model_name}.pt")

    def save_model_data(
            self,
            training_config: TrainingConfig,
            training_state: TrainingState,
            model_name: Optional[str] = None,
        ):
        if model_name is None:
            model_name = self.model_name

        model_path = ModelBase._model_path(model_name)
        pathlib.Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)

        torch.save({
            "model": {
                "class_name": type(self).__name__,
                "model_name": model_name,
                "weights": self.state_dict(),
                "config": self.config.to_dict(),
            },
            "training": {
                "config": training_config.to_dict(),
                "state": training_state.to_dict(),
            },
        }, model_path)

        print(f"Model saved to {model_path}")
    
    @classmethod
    def load(
        cls,
        model_name: str,
        override_class_name = None,
        device: Optional[str] = None,
    ) -> tuple[Self, TrainingState, TrainingConfig]:
        model_path = ModelBase._model_path(model_name)
        if device is None:
            device = select_device()

        loaded_model_data = torch.load(model_path, map_location=device)
        print(f"Model data read from {model_path}")
        loaded_class_name = loaded_model_data["model"]["class_name"]
        actual_class_name = override_class_name if override_class_name is not None else loaded_class_name

        registered_types = ModelBase.registered_types
        if actual_class_name not in registered_types:
            raise ValueError(f"Model class {actual_class_name} is not a known Model. Available classes: {list(registered_types.keys())}")
        model_class: type[Self] = registered_types[actual_class_name]

        if not issubclass(model_class, cls):
            raise ValueError(f"The model {model_name} was attempted to be loaded with {cls.__name__}.load(\"{model_name}\") (loaded class name = {loaded_class_name}, override class name = {override_class_name}), but {model_class} is not a subclass of {cls}.")

        model_name = loaded_model_data["model"]["model_name"]
        model_weights = loaded_model_data["model"]["weights"]
        model_config = model_class.config_class.from_dict(loaded_model_data["model"]["config"])
        training_config = TrainingConfig.from_dict(loaded_model_data["training"]["config"])
        training_state = TrainingState.from_dict(loaded_model_data["training"]["state"])

        model = model_class(
            model_name=model_name,
            config=model_config,
        )
        model.load_state_dict(model_weights)
        model.to(device)
        return model, training_state, training_config

    @classmethod
    def load_for_evaluation(cls, model_name: str, device: Optional[str] = None) -> Self:
        model, _, _ = cls.load(model_name=model_name, device=device)
        model.eval()
        return model

class ModelTrainerBase:
    registered_types: dict[str, type[Self]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ in ModelTrainerBase.registered_types:
            raise ValueError(f"ModelTrainer {cls.__name__} is a duplicate classname. Use a new class name.")
        ModelTrainerBase.registered_types[cls.__name__] = cls

    def __init__(
            self,
            model: ModelBase,
            config: TrainingConfig,
            continuation: Optional[TrainingState] = None,
            overrides: Optional[TrainerOverrides] = None,
        ):
        torch.manual_seed(42)

        if overrides is None:
            overrides = TrainerOverrides()

        self.model = model
        self.config = config

        self.validate_after_epochs = overrides.validate_after_epochs
        self.best_validation_loss = None
        self.early_stopping_counter = 0

        if overrides.override_to_epoch is not None:
            self.config.epochs = overrides.override_to_epoch
            print(f"Overriding training end epoch to {self.config.epochs}")

        if overrides.override_learning_rate is not None:
            print(f"Overriding learning rate to {overrides.override_learning_rate}")
            self.config.learning_rate = overrides.override_learning_rate

        match self.config.optimizer:
            case "adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            case "adamw":
                self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
            case _:
                raise ValueError(f"Unsupported optimizer type: {self.config.optimizer}")

        if continuation is not None:
            self.epoch = continuation.epoch
            self.optimizer.load_state_dict(continuation.optimizer_state)
            self.total_training_time_seconds = continuation.total_training_time_seconds
            self.latest_training_results = continuation.latest_training_results
            self.latest_validation_results = continuation.latest_validation_results
            print(f"Resuming training from epoch {self.epoch}")
        else:
            self.epoch = 0
            self.total_training_time_seconds = 0.0
            self.latest_training_results = None
            self.latest_validation_results = None

    @classmethod
    def load_with_model(cls, model_name: str, overrides: Optional[TrainerOverrides] = None, device: Optional[str] = None) -> Self:
        model, state, config = ModelBase.load(model_name=model_name, device=device)
        return cls.load(
            model=model,
            config=config,
            state=state,
            overrides=overrides,
        )

    @classmethod
    def load(cls, model: ModelBase, config: TrainingConfig, state: TrainingState, overrides: Optional[TrainerOverrides] = None) -> Self:
        trainer_class_name = state.model_trainer_class_name

        registered_types = ModelTrainerBase.registered_types
        if trainer_class_name not in registered_types:
            raise ValueError(f"Trainer class {trainer_class_name} is not a known ModelTrainer. Available classes: {list(registered_types.keys())}")
        trainer_class: type[Self] = registered_types[trainer_class_name]

        if not issubclass(trainer_class, cls):
            raise ValueError(f"The trainer was attempted to be loaded with {cls.__name__}.load(..) with a trainer class of \"{trainer_class_name}\"), but {trainer_class_name} is not a subclass of {cls}.")

        trainer = trainer_class(
            model=model,
            config=config,
            continuation=state,
            overrides=overrides,
        )

        return trainer

    def train(self) -> FullTrainingResults:
        print("Beginning training...")

        while self.epoch < self.config.epochs:
            self.epoch += 1
            print(f"Starting epoch {self.epoch}/{self.config.epochs}")
            self.train_epoch()
            print()
            if self.epoch % self.validate_after_epochs == 0 or self.epoch == self.config.epochs:
                self.run_validation()

                if wandb.run is not None:
                    log_data = {
                        "epoch": self.epoch,
                    }
                    for key, value in self.latest_validation_results.to_dict().items():
                        #note: Loss is already prefixed with "validation_"
                        log_data[f"{key}"] = value
                    for key, value in self.latest_training_results.to_dict().items():
                        if key != "epoch":
                            log_data[f"epoch_{key}"] = value
    
                    wandb.log(log_data)
                if self.config.early_stopping:
                    current_validation_loss = self.latest_validation_results.validation_loss
                    
                    if self.best_validation_loss is None or current_validation_loss < self.best_validation_loss:
                        self.best_validation_loss = current_validation_loss
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
                        if self.early_stopping_counter * self.validate_after_epochs >= self.config.early_stopping_patience:
                            print("Early stopping triggered")
                            break                
            self.save_model()
            print()

        print("Training complete.")

        return FullTrainingResults(
            total_epochs=self.config.epochs,
            last_validation=self.latest_validation_results,
            last_epoch=self.latest_training_results,
        )

    def train_epoch(self):
        self.model.train()

        # TODO: Replace with a scheduler for learning rate
        if self.epoch <= self.config.warmup_epochs:
            warmup_factor = self.epoch / self.config.warmup_epochs
            learning_rate = self.config.learning_rate * warmup_factor
            print(f"Warmup epoch {self.epoch}, learning rate set to {warmup_factor} * {self.config.learning_rate:.6f} = {learning_rate:.6f}")
        else:
            learning_rate = self.config.learning_rate

        # Apply the learning rate, even if out of warmup, to ensure warmup is disabled
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

        print_every = 10
        running_loss = 0.0
        running_samples = 0
        train_data_loader = self.get_train_data_loader()
        total_batches = len(train_data_loader)
        
        start_epoch_time_at = time.time()

        epoch_loss = 0.0
        epoch_samples = 0
        for batch_idx, raw_batch in enumerate(train_data_loader):
            self.optimizer.zero_grad()
            batch_results = self.process_batch(raw_batch)

            loss = batch_results.total_loss
            running_samples += batch_results.num_samples
            running_loss += loss.item()
            epoch_samples += batch_results.num_samples
            epoch_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            batch_num = batch_idx + 1
            if batch_num % print_every == 0:
                print(f"Epoch {self.epoch}, Batch {batch_num}/{total_batches}, Average Loss: {(running_loss / running_samples):.3g}")
                running_loss = 0.0
                running_samples = 0

        average_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        training_time = time.time() - start_epoch_time_at
        print(f"Epoch {self.epoch} complete (Average Loss: {average_loss:.3g}, Time: {training_time:.1f}s)")

        if self.total_training_time_seconds is not None:
            self.total_training_time_seconds += training_time

        self.latest_training_results = EpochTrainingResults(
            epoch=self.epoch,
            average_loss=average_loss,
            num_samples=epoch_samples,
        )

    def process_batch(self, raw_batch) -> BatchResults:
        raise NotImplementedError("This class method should be implemented by subclasses.")
    
    def get_train_data_loader(self):
        raise NotImplementedError("This class method should be implemented by subclasses.")

    def run_validation(self):
        print("== VALIDATING MODEL ==")
        print()
        self.model.eval()
        with torch.no_grad():
            validation_metrics = self._validate()
        self.latest_validation_results = validation_metrics

    def _validate(self) -> ValidationResults: 
        raise NotImplementedError("This class method should be implemented by subclasses")
    
    def save_model(self):
        training_state = TrainingState(
            epoch=self.epoch,
            optimizer_state=self.optimizer.state_dict(),
            model_trainer_class_name=self.__class__.__name__,
            total_training_time_seconds=self.total_training_time_seconds,
            latest_training_results=self.latest_training_results,
            latest_validation_results=self.latest_validation_results,
        )
        self.model.save_model_data(
            model_name=self.model.model_name,
            training_config=self.config,
            training_state=training_state
        )

        if self.latest_validation_results is None:
            print("No new validation loss available, skipping comparison with the best model.")
            return

        best_model_name = self.model.model_name + '-best'
        try:
            _, best_training_state, _ = ModelBase.load(model_name=best_model_name, device="cpu")
            best_validation_loss = best_training_state.latest_validation_results.validation_loss
            best_validation_epoch = best_training_state.latest_validation_results.epoch
        except Exception:
            best_validation_loss = None
            best_validation_epoch = None

        def format_optional_float(value):
            return f"{value:.3g}" if value is not None else "N/A"

        latest_validation_loss = self.latest_validation_results.validation_loss

        is_improvement = best_validation_loss is None or latest_validation_loss < best_validation_loss
        if is_improvement:
            print(f"The current validation loss {format_optional_float(latest_validation_loss)} is better than the previous best validation loss {format_optional_float(best_validation_loss)} from epoch {best_validation_epoch}, saving as {best_model_name}...")
            self.model.save_model_data(
                model_name=best_model_name,
                training_config=self.config,
                training_state=training_state,
            )
        else:
            print(f"The current validation loss {format_optional_float(latest_validation_loss)} is not better than the previous best validation loss {format_optional_float(best_validation_loss)} from epoch {best_validation_epoch}, so not saving as best.")

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
