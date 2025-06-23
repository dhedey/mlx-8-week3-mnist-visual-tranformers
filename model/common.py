from dataclasses import dataclass
import torch.optim as optim
import torch.nn as nn
import torch
import os
import time
import wandb
import pathlib
from typing import Optional, Self

class PersistableData:
    def to_dict(self):
        output = vars(self)
        for key, value in output.items():
            if isinstance(value, PersistableData):
                output[key] = value.to_dict()

        return output
    
    @classmethod
    def from_dict(cls, d):
        """You might need to override this method in subclasses if they have nested Persistable objects."""
        return cls(**d)

def select_device():
    DEVICE_IF_MPS_SUPPORT = 'cpu' # or 'mps' - but it doesn't work well with EmbeddingBag
    device = torch.device('cuda' if torch.cuda.is_available() else DEVICE_IF_MPS_SUPPORT if torch.backends.mps.is_available() else 'cpu')
    
    print(f'Selected device: {device}')
    return device

@dataclass
class TrainingState(PersistableData):
    epoch: int
    optimizer_state: dict
    total_training_time_seconds: Optional[float] = None
    latest_training_loss: Optional[float] = None
    latest_validation_loss: Optional[float] = None

class PersistableModel(nn.Module):
    registered_types: dict[str, type] = {}

    def __init__(self):
        super(PersistableModel, self).__init__()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.__name__ in PersistableModel.registered_types:
            raise ValueError(f"Persistable Model {cls.__name__} is a duplicate classname. Use a new class name.")
        PersistableModel.registered_types[cls.__name__] = cls

    def get_device(self):
        return next(self.parameters()).device

    @classmethod
    def build_creation_state(cls) -> dict:
        """This method should return a dictionary with the state needed to recreate the model."""
        raise NotImplementedError("This class method should be implemented by subclasses.")

    @classmethod
    def create(cls, creation_state: dict, for_evaluation_only: bool) -> Self:
        """This method should return a new model from the creation state."""
        raise NotImplementedError("This class method should be implemented by subclasses.")
    
    @staticmethod
    def _model_path(model_name: str) -> str:
        model_folder = os.path.join(os.path.dirname(__file__), "saved")
        return os.path.join(model_folder, f"{model_name}.pt")

    def save_model_data(self, model_name: str, training_state: TrainingState):
        model_path = PersistableModel._model_path(model_name)
        pathlib.Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": {
                "class_name": type(self).__name__,
                "weights": self.state_dict(),
                "creation_state": self.build_creation_state()
            },
            "training": training_state.to_dict(),
        }, model_path)
        print(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, model_name: str, for_evaluation_only: bool, override_class_name = None, device: Optional[str] = None) -> tuple[Self, TrainingState]:
        model_path = PersistableModel._model_path(model_name)
        if device is None:
            device = select_device()

        loaded_model_data = torch.load(model_path, map_location=device)
        print(f"Model data read from {model_path}")
        loaded_class_name = loaded_model_data["model"]["class_name"]
        actual_class_name = override_class_name if override_class_name is not None else loaded_class_name

        registered_types = PersistableModel.registered_types
        if actual_class_name not in registered_types:
            raise ValueError(f"Model class {actual_class_name} is not a known PersistableModel. Available classes: {list(registered_types.keys())}")
        actual_class: type[PersistableModel] = registered_types[actual_class_name]

        if not issubclass(actual_class, cls):
            raise ValueError(f"The model {model_name} was attempted to be loaded with {cls.__name__}.load(\"{model_name}\") (loaded class name = {loaded_class_name}, override class name = {override_class_name}), but {actual_class} is not a subclass of {cls}.")

        training_state = TrainingState.from_dict(loaded_model_data["training"])
        model = actual_class.create(
            creation_state=loaded_model_data["model"]["creation_state"],
            for_evaluation_only=for_evaluation_only
        )
        model.load_state_dict(loaded_model_data["model"]["weights"])
        model.to(device)
        return model, training_state

    @classmethod
    def load_for_evaluation(cls, model_name: str, device: Optional[str] = None) -> Self:
        model, _ = cls.load(model_name=model_name, device=device, for_evaluation_only=True)
        model.eval()
        return model
    
    @classmethod
    def load_for_training(cls, model_name: str, device: Optional[str] = None) -> tuple[Self, TrainingState]:
        model, training_state = cls.load(model_name=model_name, device=device, for_evaluation_only=False)
        model.train()
        return (model, training_state)


@dataclass
class TrainingHyperparameters(PersistableData):
    batch_size: int
    epochs: int
    learning_rate: float


class ModelBase(PersistableModel):
    validation_metrics: Optional[dict] = None

    """A base class for all our dual encoder models."""
    def __init__(self, model_name: str, training_parameters: TrainingHyperparameters, model_parameters: PersistableData):
        super(ModelBase, self).__init__()
        self.model_name = model_name
        self.training_parameters = training_parameters
        self.model_parameters=model_parameters

    def build_creation_state(self) -> dict:
        return {
            "model_name": self.model_name,
            "hyper_parameters": self.model_parameters.to_dict(),
            "training_parameters": self.training_parameters.to_dict(),
            "validation_metrics": self.validation_metrics,
        }
    
    def set_validation_metrics(self, validation_metrics: dict):
        """Set the validation metrics for the model."""
        self.validation_metrics = validation_metrics

    @classmethod
    def create(cls, creation_state: dict, for_evaluation_only: bool) -> Self:
        """This method should return a new model from the creation state."""
        model = cls(
            model_name=creation_state["model_name"],
            training_parameters=TrainingHyperparameters.from_dict(creation_state["training_parameters"]),
            model_parameters=cls.hyper_parameters_class().from_dict(creation_state["hyper_parameters"]),
        )
        model.validation_metrics = creation_state.get("validation_metrics", None)
        return model
    
    @classmethod
    def hyper_parameters_class(cls) -> type[PersistableData]:
        raise NotImplementedError("This method should be implemented by subclasses.")


class ModelTrainerBase:
    def __init__(
            self,
            model: ModelBase,
            continuation: Optional[TrainingState] = None,
            override_to_epoch: Optional[int] = None,
            validate_after_epochs: int = 5,
            immediate_validation: bool = False,
        ):
        torch.manual_seed(42)

        self.model = model
        self.validate_and_save_after_epochs = validate_after_epochs

        self.optimizer = optim.Adam(self.model.parameters(), lr=model.training_parameters.learning_rate)

        if continuation is not None:
            self.epoch = continuation.epoch
            self.optimizer.load_state_dict(continuation.optimizer_state)
            self.total_training_time_seconds = continuation.total_training_time_seconds
            self.latest_training_loss = continuation.latest_training_loss
            self.latest_validation_loss = continuation.latest_validation_loss
            print(f"Resuming training from epoch {self.epoch}")
        else:
            self.epoch = 0
            self.total_training_time_seconds = 0.0
            self.latest_training_loss = None
            self.latest_validation_loss = None

        if override_to_epoch is not None:
            self.model.training_parameters.epochs = override_to_epoch
            print(f"Overriding training end epoch to {self.model.training_parameters.epochs}")

        print("Preparing datasets...")

        # TODO!

        if immediate_validation:
            print("Immediate validation requested, validating model...")
            self.validate()

    def train(self):
        print("Beginning training...")

        last_epoch_results = None

        while self.epoch < self.model.training_parameters.epochs:
            self.epoch += 1
            print(f"Starting epoch {self.epoch}/{self.model.training_parameters.epochs}")
            last_epoch_results = self.train_epoch()
            if self.epoch % self.validate_and_save_after_epochs == 0 or self.epoch == self.model.training_parameters.epochs:
                validation_metrics = self.validate()

                if wandb.run is not None:
                    log_data = {
                        "epoch": self.epoch,
                    }
                    for key, value in validation_metrics.items():
                        log_data[f"validation_{key}"] = value
                    for key, value in last_epoch_results.items():
                        log_data[f"epoch_{key}"] = value
    
                    wandb.log(log_data)

            self.save_model()

        print("Training complete.")

        return {
            "total_epochs": self.model.training_parameters.epochs,
            "validation": self.model.validation_metrics,
            "last_epoch": last_epoch_results,
        }

    def train_epoch(self):
        self.model.train()

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
            batch_results = self.process_test_batch(raw_batch)

            loss = batch_results["total_loss"]
            running_samples += batch_results["num_samples"]
            running_loss += loss.item()
            epoch_loss += loss.item()
            epoch_samples += batch_results["num_samples"]

            loss.backward()
            self.optimizer.step()

            batch_num = batch_idx + 1
            if batch_num % print_every == 0:
                print(f"Epoch {self.epoch}, Batch {batch_num}/{total_batches}, Average Loss: {(running_loss / running_samples):.4f}")
                running_loss = 0.0
                running_samples = 0

        average_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
        training_time = time.time() - start_epoch_time_at
        print(f"Epoch {self.epoch} complete (Average Loss: {average_loss:.4f}, Time: {training_time:.1f}s)")

        if self.total_training_time_seconds is not None:
            self.total_training_time_seconds += training_time

        self.latest_training_loss = average_loss

        return {
            "average_loss": average_loss,
        }

    def process_test_batch(self, raw_batch) -> dict:
        raise NotImplementedError("This class method should be implemented by subclasses.")
    
    def get_train_data_loader(self):
        raise NotImplementedError("This class method should be implemented by subclasses.")

    def validate(self):
        print()
        print("== VALIDATING MODEL ==")
        print()
        self.model.eval()
        validation_metrics = self._validate()
        self.model.set_validation_metrics(validation_metrics)
        return validation_metrics

    def _validate(self) -> dict: 
        raise NotImplementedError("This class method should be implemented by subclasses")
    
    def save_model(self):
        training_state = TrainingState(
            epoch=self.epoch,
            optimizer_state=self.optimizer.state_dict(),
            total_training_time_seconds=self.total_training_time_seconds,
            latest_training_loss=self.latest_training_loss,
            latest_validation_loss=self.latest_validation_loss,
        )
        self.model.save_model_data(
            model_name=self.model.model_name,
            training_state=training_state
        )
        best_model_name = self.model.model_name + '-best'
        try:
            _, best_training_state = ModelBase.load_for_training(model_name=best_model_name)
            best_validation_loss = best_training_state.latest_validation_loss
            best_validation_epoch = best_training_state.epoch
        except Exception:
            best_validation_loss = None
            best_validation_epoch = None

        def format_optional_float(value):
            return f"{value:.2f}" if value is not None else "N/A"

        is_improvement = self.latest_validation_loss is not None and (best_validation_loss is None or self.latest_validation_loss < best_validation_loss)
        if is_improvement:
            print(f"The current validation loss {format_optional_float(self.latest_validation_loss)} is better than the previous best validation loss {format_optional_float(best_validation_loss)} from epoch {best_validation_epoch}, saving as {best_model_name}...")
            self.model.save_model_data(model_name=best_model_name, training_state=training_state)
        else:
            print(f"The current validation loss {format_optional_float(self.latest_validation_loss)} is not better than the previous best validation loss {format_optional_float(best_validation_loss)} from epoch {best_validation_epoch}, so not saving as best.")
