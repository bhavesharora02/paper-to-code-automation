import pytest
import json
from pathlib import Path
from scripts.ir_generation.model_ir import (
    ModelIR, ModelArchitecture, LayerSpec, 
    DatasetSpec, TrainingConfig, OptimizerSpec,
    ModelType, LayerType, OptimizerType, LossType,
    create_default_ir
)
from scripts.ir_generation.ir_generator import IRGenerator

# Test data
SAMPLE_PAPER_TEXT = """
Title: Test Transformer Model
Authors: Test Author 1, Test Author 2

Abstract: This is a test paper describing a transformer model.

Model Architecture:
The model consists of an embedding layer followed by 6 transformer layers.
Each transformer has 8 attention heads and a hidden size of 512.

Training:
We trained the model using Adam optimizer with a learning rate of 0.0001.
The batch size was set to 32 and we trained for 100 epochs.

Results:
The model achieved an accuracy of 90.5% on the test set.
"""

# Fixtures
@pytest.fixture
def sample_ir() -> ModelIR:
    """Create a sample IR for testing."""
    return create_default_ir()

@pytest.fixture
def ir_generator() -> IRGenerator:
    """Create an IRGenerator instance for testing."""
    return IRGenerator()

# Tests
def test_create_default_ir(sample_ir):
    """Test that a default IR is created with expected defaults."""
    assert sample_ir.model is not None
    assert sample_ir.training is not None
    assert sample_ir.model.type == ModelType.NLP
    assert len(sample_ir.model.layers) == 0
    assert sample_ir.training.optimizer.type == OptimizerType.ADAM

def test_ir_generator_metadata(ir_generator):
    """Test that metadata is correctly extracted from text."""
    ir_generator._extract_metadata(SAMPLE_PAPER_TEXT)
    
    assert "Test Transformer Model" in ir_generator.ir.paper_metadata.title
    assert "Test Author 1" in ir_generator.ir.paper_metadata.authors

def test_ir_generator_architecture(ir_generator):
    """Test that model architecture is correctly extracted."""
    ir_generator._extract_architecture(SAMPLE_PAPER_TEXT)
    
    # Should find at least the embedding and transformer layers
    assert len(ir_generator.ir.model.layers) >= 1
    assert any(layer.type == LayerType.TRANSFORMER for layer in ir_generator.ir.model.layers)

def test_ir_generator_training(ir_generator):
    """Test that training configuration is correctly extracted."""
    ir_generator._extract_training_config(SAMPLE_PAPER_TEXT)
    
    assert ir_generator.ir.training.optimizer.type == OptimizerType.ADAM
    assert ir_generator.ir.training.learning_rate == 0.0001
    assert ir_generator.ir.training.batch_size == 32

def test_ir_generator_evaluation(ir_generator):
    """Test that evaluation metrics are correctly extracted."""
    ir_generator._extract_evaluation_metrics(SAMPLE_PAPER_TEXT)
    
    assert "accuracy" in ir_generator.ir.evaluation
    assert abs(ir_generator.ir.evaluation["accuracy"] - 90.5) < 0.1

def test_generate_ir_from_text(tmp_path):
    """Test the complete IR generation pipeline."""
    output_path = tmp_path / "test_ir.json"
    ir = IRGenerator().from_extracted_text(SAMPLE_PAPER_TEXT)
    
    # Save and load to test serialization
    with open(output_path, 'w') as f:
        json.dump(ir.dict(), f)
    
    with open(output_path, 'r') as f:
        loaded_ir = ModelIR(**json.load(f))
    
    assert loaded_ir.model is not None
    assert loaded_ir.training is not None
    assert isinstance(loaded_ir, ModelIR)

def test_layer_spec_validation():
    """Test that layer specifications are properly validated."""
    # Valid layer
    valid_layer = LayerSpec(
        name="test_layer",
        type=LayerType.LINEAR,
        parameters={"in_features": 512, "out_features": 256},
        input_shape=[32, 512],
        output_shape=[32, 256]
    )
    assert valid_layer.name == "test_layer"
    
    # Invalid layer type should raise ValueError
    with pytest.raises(ValueError):
        LayerSpec(
            name="invalid_layer",
            type="invalid_type",
            parameters={}
        )

# Run tests with: pytest tests/test_ir_generation.py -v