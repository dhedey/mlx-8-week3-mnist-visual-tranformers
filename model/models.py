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
from .trainer import EncoderOnlyModelTrainer, TrainerParameters

@dataclass
class TransformerEncoderModelHyperparameters(PersistableData):
    encoder_blocks: int
    embedding_size: int
    kq_dimension: int
    v_dimension: int
    mlp_hidden_dimension: int
    heads_per_layer: int = field(default=1, metadata={"description": "Number of attention heads per encoder block."})
    add_positional_bias: bool = field(default=False, metadata={"description": "Whether to add a positional bias to the block embeddings."})
    mlp_dropout: float = field(default=0.0, metadata={"description": "Dropout rate for the MLP layers."})

class TransformerEncoderModel(ModelBase):
    def __init__(self, model_name: str, training_parameters: TrainingHyperparameters, model_parameters: TransformerEncoderModelHyperparameters):
        super(TransformerEncoderModel, self).__init__(
            model_name=model_name,
            training_parameters=training_parameters,
            model_parameters=model_parameters,
        )
        self.image_block_embedding = nn.Linear(
            in_features=7 * 7, # Each block is 7x7 pixels
            out_features=model_parameters.embedding_size,
        )
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                kq_dimension=model_parameters.kq_dimension,
                v_dimension=model_parameters.v_dimension,
                embedding_dimension=model_parameters.embedding_size,
                mlp_hidden_dimension=model_parameters.mlp_hidden_dimension,
                mlp_dropout=model_parameters.mlp_dropout,
                attention_heads=model_parameters.heads_per_layer,
            )
            for _ in range(model_parameters.encoder_blocks)
        ])
        self.prediction_layer = nn.Linear(model_parameters.embedding_size, 10)
        # Allow the model to learn a positional bias for each of the blocks in the image
        self.positional_bias = nn.Parameter(
            torch.zeros([4 * 4, model_parameters.embedding_size], dtype=torch.float32),
            requires_grad=model_parameters.add_positional_bias,
        )
        
    @classmethod
    def hyper_parameters_class(cls) -> type[PersistableData]:
        return TransformerEncoderModelHyperparameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Assumes it receives a (Batch(es), ChannelCount=1, Height=28, Width=28) tensor of images."""
        #Move to device if not already there
        x = x.to(self.get_device())
        
        # Unfold into 7x7 blocks
        x = torch.nn.Unfold(       # Shape: (Batch(es), Channel=1*(7*7), NumBlocks = 4*4)
            kernel_size=(7, 7),
            stride=(7, 7),
            padding=0,
        )(x)
        x = x.transpose(-1, -2)    # Shape: (Batch(es), NumBlocks = 16, Channel=1*(7*7))

        residual_stream = self.image_block_embedding(x) # Shape: (Batch(es), NumBlocks = 16, Embedding Dimension)
        residual_stream = residual_stream + self.positional_bias

        for encoder_block in self.encoder_blocks:
            residual_stream = encoder_block(residual_stream)

        averaged_embedding = residual_stream.mean(dim=-2)        # Average over the blocks, Shape: (Batch(es), Embedding Dimension)
        logits = self.prediction_layer(averaged_embedding)  # Shape: (Batch(es), 10)

        return logits

class EncoderBlock(nn.Module):
    def __init__(
            self,
            kq_dimension:int,
            v_dimension: int,
            embedding_dimension: int,
            mlp_dropout: float,
            attention_heads: int,
            mlp_hidden_dimension: Optional[int] = None,
        ):
        super().__init__()
        attention_heads = [
            UnmaskedAttentionHead(
                kq_dimension=kq_dimension,
                v_dimension=v_dimension,
                encoder_embedding_dimension=embedding_dimension,
                decoder_embedding_dimension=embedding_dimension,
            )
            for _ in range(attention_heads)
        ]
        # Backwards compatibility
        if len(attention_heads) == 1:
            self.attention_head = attention_heads[0]
            self.attention_heads = None
        else:
            self.attention_heads = nn.ModuleList(attention_heads)
            self.attention_head = None
        
        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        if mlp_hidden_dimension is None:
            mlp_hidden_dimension = embedding_dimension * 4 # Commonly used in transformers
        self.mlp = MultiLayerPerceptron(
            embedding_dimension=embedding_dimension,
            hidden_dimension=embedding_dimension * 4,
            dropout=mlp_dropout,
        )
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)

    def forward(self, residual_stream):
        # Shape: (Batch, Sequence Length, Embedding Dimension)
        attention_heads = self.attention_heads if self.attention_heads is not None else [self.attention_head]
        attention_head_results = [head(residual_stream, residual_stream) for head in attention_heads]
        for result in attention_head_results:
            residual_stream = residual_stream + result
        residual_stream = self.layer_norm_1(residual_stream)
        residual_stream = residual_stream + self.mlp(residual_stream)         
        residual_stream = self.layer_norm_2(residual_stream)

        return residual_stream
    
class MultiLayerPerceptron(nn.Module):
    def __init__(self, embedding_dimension: int, hidden_dimension: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, embedding_dimension)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x has            Shape: (batch_size, sequence_length, embedding_dimension)
        x = self.fc1(x)  # Shape: (batch_size, sequence_length, hidden_dimension)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # Shape: (batch_size, sequence_length, embedding_dimension)
        return x
    
class UnmaskedAttentionHead(nn.Module):
    def __init__(self, kq_dimension, v_dimension, encoder_embedding_dimension, decoder_embedding_dimension):
        super().__init__()
        self.kq_dimension = kq_dimension
        self.v_dimension = v_dimension
        self.encoder_embedding_dimension = encoder_embedding_dimension
        self.decoder_embedding_dimension = decoder_embedding_dimension

        self.query_projection = nn.Linear(decoder_embedding_dimension, kq_dimension)
        self.key_projection = nn.Linear(encoder_embedding_dimension, kq_dimension)
        self.value_projection = nn.Linear(encoder_embedding_dimension, v_dimension)
        
        # NB - Unlike many transformer implementations, we have a per-head output projection for easy of understanding.
        self.output_projection = nn.Linear(v_dimension, decoder_embedding_dimension)

    def forward(self, encoder_embeddings, decoder_embeddings):
        queries = self.query_projection(decoder_embeddings)  # Shape: (batch_size, decoder_sequence_length, kq_dimension)
        keys = self.key_projection(encoder_embeddings)       # Shape: (batch_size, encoder_sequence_length, kq_dimension)
        values = self.value_projection(encoder_embeddings)   # Shape: (batch_size, encoder_sequence_length, v_dimension)

        attention_scores = queries @ keys.transpose(-2, -1)  # Shape: (batch_size, decoder_sequence_length, encoder_sequence_length)

        # Softmax over the encoder_sequence_length (keys), so each query row sums to 1
        attention = F.softmax(attention_scores / math.sqrt(self.kq_dimension), dim=-1)

        output_values = attention @ values                   # Shape: (batch_size, decoder_sequence_length, v_dimension)
        
        residual = self.output_projection(output_values)     # Shape: (batch_size, decoder_sequence_length, decoder_embedding_dimension)
        return residual

class MaskedSelfAttention(nn.Module):
    def __init__(self, embedding_dimension, kq_dimension, v_dimension, num_heads, mask_future_tokens=True):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.kq_dimension = kq_dimension
        self.v_dimension = v_dimension
        self.num_heads = num_heads
        self.query_projection = nn.Linear(embedding_dimension, kq_dimension * num_heads)
        self.key_projection = nn.Linear(embedding_dimension, kq_dimension * num_heads)
        self.value_projection = nn.Linear(embedding_dimension, v_dimension * num_heads)
        self.output_projection = nn.Linear(v_dimension * num_heads, embedding_dimension)
        self.mask_future_tokens = mask_future_tokens
        self._cached_mask = None
        self._cached_mask_device = None
        self._cached_mask_sequence_length = None


    def forward(self, x): #x has shape (batch_size, sequence_length, embedding_dimension)
        queries = self.query_projection(x).reshape(*x.shape[:-2], self.num_heads, self.kq_dimension).transpose(-2, -3)  # Shape: (batch_size, num_heads, sequence_length, kq_dimension)
        keys = self.key_projection(x).reshape(*x.shape[:-2], self.num_heads, self.kq_dimension).transpose(-2, -3)       # Shape: (batch_size, num_heads, sequence_length, kq_dimension)
        values = self.value_projection(x).reshape(*x.shape[:-2], self.num_heads, self.v_dimension).transpose(-2, -3)   # Shape: (batch_size, num_heads, sequence_length, v_dimension)

        attention_scores = queries @ keys.transpose(-2, -1) # Shape: (batch_size, num_heads, sequence_length, sequence_length)
        if self.mask_future_tokens:
            if self._cached_mask is None or self._cached_mask_device != x.device or self._cached_mask_sequence_length != x.shape[-2]:
            # Create a causal mask for the attention scores (ones indicate positions to mask    )
                self._cached_mask = torch.triu(torch.ones(x.shape[-2], x.shape[-2]), diagonal=1, dtype=torch.bool, device=x.device)
                self._cached_mask_sequence_length = x.shape[-2] 
                self._cached_mask_device = x.device
            attention_scores = attention_scores.masked_fill(self._cached_mask, float("-inf"))

        attention = F.softmax(attention_scores / math.sqrt(self.kq_dimension), dim=-1)
        output_values = attention @ values                   # Shape: (batch_size, num_heads, sequence_length, v_dimension)
        output_values = output_values.transpose(-2, -3).reshape(*x.shape[:-2], self.num_heads * self.v_dimension) # Shape: (batch_size, sequence_length, num_heads * v_dimension)
        residual = self.output_projection(output_values)     # Shape: (batch_size, sequence_length, embedding_dimension)
        return residual

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
    "encoder-only-bigger": {
        "training": TrainingHyperparameters(
            batch_size=256,
            epochs=50,
            learning_rate=0.0005,
            optimizer="adamw",
            warmup_epochs=5,
        ),
        "model": TransformerEncoderModelHyperparameters(
            encoder_blocks=5,
            embedding_size=64,
            kq_dimension=16,
            v_dimension=16,
            mlp_hidden_dimension=256,
            mlp_dropout=0.3,
            add_positional_bias=True,
            heads_per_layer=2,
        ),
        "model_class": TransformerEncoderModel,
        "model_trainer": EncoderOnlyModelTrainer,
    },
    "encoder-only-big": {
        "training": TrainingHyperparameters(
            batch_size=256,
            epochs=50,
            learning_rate=0.002,
            optimizer="adamw",
            warmup_epochs=5,
        ),
        "model": TransformerEncoderModelHyperparameters(
            encoder_blocks=5,
            embedding_size=32,
            kq_dimension=16,
            v_dimension=16,
            mlp_hidden_dimension=128,
            mlp_dropout=0.3,
            add_positional_bias=True,
            heads_per_layer=1,
        ),
        "model_class": TransformerEncoderModel,
        "model_trainer": EncoderOnlyModelTrainer,
    },
    "encoder-only-positional-dropout": {
        "training": TrainingHyperparameters(
            batch_size=256,
            epochs=50,
            learning_rate=0.002,
            optimizer="adamw",
            warmup_epochs=5,
        ),
        "model": TransformerEncoderModelHyperparameters(
            encoder_blocks=3,
            embedding_size=32,
            kq_dimension=16,
            v_dimension=16,
            mlp_hidden_dimension=128, # 4 * embedding_size is typical in transformers
            mlp_dropout=0.3,
            add_positional_bias=True,
        ),
        "model_class": TransformerEncoderModel,
        "model_trainer": EncoderOnlyModelTrainer,
    },
    "encoder-only-positional": {
        "training": TrainingHyperparameters(
            batch_size=128,
            epochs=20,
            learning_rate=0.002,
        ),
        "model": TransformerEncoderModelHyperparameters(
            encoder_blocks=3,
            embedding_size=32,
            kq_dimension=16,
            v_dimension=16,
            mlp_hidden_dimension=128, # 4 * embedding_size is typical in transformers
            add_positional_bias=True,
        ),
        "model_class": TransformerEncoderModel,
        "model_trainer": EncoderOnlyModelTrainer,
    },
    "encoder-only": {
        "training": TrainingHyperparameters(
            batch_size=128,
            epochs=20,
            learning_rate=0.002,
        ),
        "model": TransformerEncoderModelHyperparameters(
            encoder_blocks=3,
            embedding_size=32,
            kq_dimension=16,
            v_dimension=16,
            mlp_hidden_dimension=128, # 4 * embedding_size is typical in transformers
            add_positional_bias=False,
        ),
        "model_class": TransformerEncoderModel,
        "model_trainer": EncoderOnlyModelTrainer,
    },
}

        
if __name__ == "__main__":
   for model_name, parameters in DEFAULT_MODEL_PARAMETERS.items():
        best_version = f"{model_name}-best"
        print(f"Loading Model: {best_version}")
        model = ModelBase.load_for_evaluation(best_version)
        print(f"Validation Metrics: {model.validation_metrics}")
   
        # Re-validate the model to check the loading has worked correctly
        trainer = parameters["model_trainer"](
            model=model,
            parameters=TrainerParameters()
        )
        trainer.validate()