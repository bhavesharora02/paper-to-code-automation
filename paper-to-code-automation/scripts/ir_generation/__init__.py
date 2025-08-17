"""
IR Generation Module

This module provides functionality to generate an Intermediate Representation (IR)
from research paper text. It includes the IR schema, generator, and related utilities.
"""

# Import main components to make them available at the package level
from .model_ir import (
    ModelIR,
    ModelArchitecture,
    LayerSpec,
    DatasetSpec,
    TrainingConfig,
    OptimizerSpec,
    ModelType,
    LayerType,
    OptimizerType,
    LossType,
    create_default_ir
)

from .ir_generator import IRGenerator, generate_ir_from_text

__all__ = [
    'ModelIR',
    'ModelArchitecture',
    'LayerSpec',
    'DatasetSpec',
    'TrainingConfig',
    'OptimizerSpec',
    'ModelType',
    'LayerType',
    'OptimizerType',
    'LossType',
    'create_default_ir',
    'IRGenerator',
    'generate_ir_from_text'
]
