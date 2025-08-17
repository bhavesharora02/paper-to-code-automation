import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import json

from scripts.ir_generation.llm_extractor import LLMExtractor
from scripts.ir_generation.model_ir import ModelIR, ModelArchitecture, LayerSpec, TrainingConfig, OptimizerSpec

# Sample paper text for testing
SAMPLE_PAPER = """
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

# Expected IR structure based on the sample paper
EEXPECTED_IR = {
    "paper_metadata": {
        "title": "Test Transformer Model",
        "authors": ["Test Author 1", "Test Author 2"],
        "year": 2023,
        "doi": None,
        "url": None,
        "created_at": "2023-01-01T00:00:00"
    },
    "model": {
        "name": "test_transformer",
        "type": "transformer",
        "layers": [
            {
                "name": "embedding",
                "type": "embedding",
                "parameters": {"dim": 512}
            },
            {
                "name": "transformer_layer_1",
                "type": "transformer",
                "parameters": {
                    "num_heads": 8,
                    "hidden_size": 512
                }
            }
        ]
    },
    "training": {
        "optimizer": {
            "type": "adam",
            "parameters": {"learning_rate": 0.0001}
        },
        "batch_size": 32,
        "epochs": 100,
        "loss": "cross_entropy"  # Changed from loss_function to loss
    },
    "dataset": {
        "name": None,
        "source": None,
        "size": None,
        "split": None
    },
    "evaluation": {
        "accuracy": 90.5
    }
}


@pytest.fixture
def llm_extractor():
    """Create an LLMExtractor instance with a mock LLM."""
    with patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_llm:
        extractor = LLMExtractor()
        mock_llm.return_value = MagicMock()
        yield extractor

@patch('langchain_core.prompts.ChatPromptTemplate.from_messages')
@patch('langchain_core.output_parsers.JsonOutputParser')
def test_extract_ir(mock_parser, mock_prompt, llm_extractor):
    """Test that extract_ir returns a valid ModelIR object."""
    # Setup mock parser
    mock_parser.return_value = MagicMock()
    
    # Setup mock prompt and chain
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = EXPECTED_IR
    mock_prompt.return_value = MagicMock()
    mock_prompt.return_value.pipe.return_value = mock_chain
    
    # Call the method
    result = llm_extractor.extract_ir(SAMPLE_PAPER)
    
    # Verify the result
    assert isinstance(result, ModelIR)
    assert result.paper_metadata.title == "Test Transformer Model"
    assert len(result.model.layers) == 2
    assert result.model.layers[0].type == "embedding"
    assert result.model.layers[1].type == "transformer"
    assert result.training.optimizer.type == "adam"
    assert result.training.loss == "cross_entropy"  # Updated to use loss instead of loss_function
    assert result.evaluation.get("accuracy") == 90.5

def test_convert_to_ir():
    """Test the _convert_to_ir method with sample data."""
    extractor = LLMExtractor()
    ir = extractor._convert_to_ir(EXPECTED_IR)
    
    # Verify the conversion
    assert isinstance(ir, ModelIR)
    assert ir.paper_metadata.title == "Test Transformer Model"
    assert len(ir.model.layers) == 2  # embedding + transformer layer
    assert ir.training.batch_size == 32
    assert ir.training.epochs == 100
    assert ir.evaluation["accuracy"] == 90.5
    assert ir.model.layers[0].parameters["dim"] == 512
    assert ir.model.layers[1].parameters["num_heads"] == 8
    assert ir.model.layers[1].parameters["hidden_size"] == 512

@patch('langchain_core.prompts.ChatPromptTemplate.from_messages')
def test_extract_ir_error_handling(mock_prompt, llm_extractor):
    """Test that extract_ir handles errors gracefully."""
    # Setup mock to raise an exception
    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = Exception("API Error")
    mock_prompt.return_value = MagicMock()
    mock_prompt.return_value.pipe.return_value = mock_chain
    
    # Call the method
    result = llm_extractor.extract_ir(SAMPLE_PAPER)
    
    # Should return a default ModelIR on error
    assert isinstance(result, ModelIR)
    assert result.paper_metadata.title == ""  # Default empty title
    assert len(result.model.layers) == 0  # No layers in default IR