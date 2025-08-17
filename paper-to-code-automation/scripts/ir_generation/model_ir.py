from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, validator
from datetime import datetime

class PaperMetadata(BaseModel):
    """Metadata about the research paper."""
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    url: Optional[str] = None
    doi: Optional[str] = None

class ModelType(str, Enum):
    NLP = "nlp"
    CV = "cv"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"

class LayerType(str, Enum):
    EMBEDDING = "embedding"
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    GRU = "gru"
    LINEAR = "linear"
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    CONV3D = "conv3d"
    DROPOUT = "dropout"
    LAYER_NORM = "layer_norm"
    BATCH_NORM = "batch_norm"
    ACTIVATION = "activation"
    POOLING = "pooling"
    FLATTEN = "flatten"
    RESIDUAL = "residual"
    ATTENTION = "attention"

class OptimizerType(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    ADAMAX = "adamax"
    NADAM = "nadam"

class LossType(str, Enum):
    CROSS_ENTROPY = "cross_entropy"
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    COSINE_SIMILARITY = "cosine_similarity"
    TRIPLET = "triplet"
    FOCAL = "focal"
    DICE = "dice"
    BCE_WITH_LOGITS = "bce_with_logits"

class LayerSpec(BaseModel):
    """Specification for a single layer in the model."""
    name: str
    type: LayerType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None

class DatasetSpec(BaseModel):
    """Specification for the dataset used in training."""
    name: str
    input_shape: List[Union[int, str]]
    output_shape: List[Union[int, str]]
    preprocessing: Dict[str, Any] = Field(default_factory=dict)
    source: Optional[str] = None

class OptimizerSpec(BaseModel):
    """Specification for the optimizer."""
    type: OptimizerType
    parameters: Dict[str, Any] = Field(default_factory=dict)

class TrainingConfig(BaseModel):
    """Training configuration."""
    dataset: DatasetSpec = Field(default_factory=lambda: DatasetSpec(
        name="default_dataset",
        input_shape=["batch_size", "seq_len"],
        output_shape=["batch_size", "num_classes"]
    ))
    batch_size: int = 32
    epochs: int = 10
    optimizer: OptimizerSpec = Field(default_factory=lambda: OptimizerSpec(
        type=OptimizerType.ADAM,
        parameters={"lr": 0.001}
    ))
    loss: LossType = LossType.CROSS_ENTROPY
    metrics: List[str] = Field(default_factory=list)
    learning_rate: float = 0.001

class ModelArchitecture(BaseModel):
    """Model architecture specification."""
    name: str = "default_model"
    type: ModelType = ModelType.NLP
    layers: List[LayerSpec] = Field(default_factory=list)
    input_spec: Dict = Field(default_factory=dict)
    output_spec: Dict = Field(default_factory=dict)
    description: Optional[str] = None

def create_default_ir() -> 'ModelIR':
    """Create a default ModelIR instance with sensible defaults."""
    return ModelIR()

class ModelIR(BaseModel):
    """Complete Intermediate Representation of a model from a research paper."""
    # Paper metadata
    paper_metadata: PaperMetadata = Field(default_factory=PaperMetadata)
    
    # Model architecture
    model: ModelArchitecture = Field(default_factory=ModelArchitecture)
    
    # Training configuration
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    
    # Evaluation metrics
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0.0"