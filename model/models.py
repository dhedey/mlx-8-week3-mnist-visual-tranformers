from dataclasses import dataclass, field
import torch.nn.functional as F
import torch.nn as nn
import torch
import re
import os
import statistics
import transformers
import random
import pandas as pd
import math
from typing import Optional, Self
from .common import ModelBase, PersistableData, TrainingHyperparameters
from .trainer import EncoderOnlyModelTrainer

@dataclass
class TransformerBasedEncoderModelHyperparameters(PersistableData):
    transformer_depth: int
    embedding_size: int
    # TODO

class TransformerBasedEncoderModel(ModelBase):
    def __init__(self, model_name: str, training_parameters: TrainingHyperparameters, model_parameters: TransformerBasedEncoderModelHyperparameters):
        super(TransformerBasedEncoderModel, self).__init__(
            model_name=model_name,
            training_parameters=training_parameters,
            model_parameters=model_parameters,
        )

        # TODO
    
    @classmethod
    def hyper_parameters_class(cls) -> type[PersistableData]:
        return TransformerBasedEncoderModelHyperparameters

class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size, include_layer_norm, dropout):
        super(HiddenLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        if include_layer_norm:
            self.layer_norm = nn.LayerNorm(output_size)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x

DEFAULT_MODEL_PARAMETERS = {
    "encoder-only": {
        "training": TrainingHyperparameters(
            batch_size=128,
            epochs=20,
            learning_rate=0.002,
        ),
        "model": TransformerBasedEncoderModelHyperparameters(
            transformer_depth=3,
            embedding_size=32,
        ),
        "model_class": TransformerBasedEncoderModel,
        "model_trainer": EncoderOnlyModelTrainer,
    }
}

        
if __name__ == "__main__":
    raise NotImplementedError("Add something here which can test existing model/s.")