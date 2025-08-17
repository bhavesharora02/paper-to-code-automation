"""
LLM-based extractor for generating IR from research papers.
Uses LangChain and Google's Gemini to extract structured information from paper text.
"""
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .model_ir import ModelIR, ModelArchitecture, LayerSpec, DatasetSpec, TrainingConfig, OptimizerSpec, create_default_ir

# Load environment variables
load_dotenv()

class LLMExtractor:
    """Extracts IR from paper text using Google's Gemini model."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro", temperature: float = 0.1):
        """Initialize the LLM extractor.
        
        Args:
            model_name: Name of the Gemini model to use (default: gemini-1.5-pro)
            temperature: Temperature for generation (lower = more deterministic)
        """
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.parser = JsonOutputParser()
        
    def extract_ir(self, paper_text: str) -> ModelIR:
        """Extract IR from paper text using Gemini.
        
        Args:
            paper_text: Raw text from the research paper
            
        Returns:
            ModelIR: The extracted IR
        """
        # Define the schema for the LLM to follow
        schema = {
            "paper_metadata": {
                "title": "string",
                "authors": ["string"],
                "year": "integer",
                "doi": "string or null",
                "url": "string or null"
            },
            "model": {
                "name": "string",
                "type": "string",
                "layers": [{
                    "name": "string",
                    "type": "string",
                    "parameters": {}
                }]
            },
            "training": {
                "optimizer": {
                    "type": "string",
                    "parameters": {"learning_rate": "float"}
                },
                "batch_size": "integer",
                "epochs": "integer",
                "loss": "string"  # Changed from loss_function to loss
            },
            "dataset": {
                "name": "string or null",
                "source": "string or null",
                "size": "integer or null",
                "split": "dict or null"
            },
            "evaluation": {"accuracy": "float"}
        }
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting structured information from AI research papers. 
            Extract the following information from the provided paper text and format it as JSON.
            Only include information that is explicitly mentioned in the text.
            
            Schema to follow:
            {schema}
            
            Instructions:
            - For authors: Format as a list of strings, each containing a full name
            - For layers: Include all layers mentioned in the architecture
            - For parameters: Include all numerical parameters mentioned
            - For evaluation: Include all metrics with their values
            - If a field is not mentioned, set it to null
            
            Return only the JSON object, no other text."""),
            ("human", "Paper text: {paper_text}")
        ])
        
        try:
            # Create the chain
            chain = prompt.pipe(self.llm).pipe(self.parser)
            
            # Get the LLM response
            result = chain.invoke({
                "schema": json.dumps(schema, indent=2),
                "paper_text": paper_text[:10000]  # Limit context length
            })
            
            # Convert to ModelIR
            return self._convert_to_ir(result)
            
        except Exception as e:
            print(f"Error during Gemini extraction: {str(e)}")
            # Fall back to default IR
            return create_default_ir()
    def _map_layer_type(self, layer_type: str) -> str:
        """Map common layer type names to supported values."""
        layer_mapping = {
            'convolutional': 'conv2d',
            'conv': 'conv2d',
            'cnn': 'conv2d',
            'convolution': 'conv2d',
            'dense': 'linear',
            'fully-connected': 'linear',  # Added mapping for 'fully-connected'
            'fully_connected': 'linear',
            'fc': 'linear',
            'rnn': 'lstm',  # Default RNN to LSTM
            'recurrent': 'lstm',
            'normalization': 'batch_norm',
            'batchnorm': 'batch_norm',
            'batchnormalization': 'batch_norm',
            'layernorm': 'layer_norm',
            'layernormalization': 'layer_norm',
            'maxpool': 'pooling',
            'maxpooling': 'pooling',
            'avgpool': 'pooling',
            'averagepooling': 'pooling',
            'globalpooling': 'pooling',
            'relu': 'activation',
            'sigmoid': 'activation',
            'tanh': 'activation',
            'softmax': 'activation',
            'attention_layer': 'attention',
            'self_attention': 'attention',
            'transformer_layer': 'transformer',
            'transformer_encoder': 'transformer',
            'transformer_decoder': 'transformer',
        }
        return layer_mapping.get(layer_type.lower(), layer_type)

    def _convert_to_ir(self, data: Dict[str, Any]) -> ModelIR:
        """Convert the LLM output to a ModelIR object with enhanced error handling."""
        try:
            # Start with a default IR
            ir = create_default_ir()
            
            # Debug: Print the raw data we're processing
            print("\nRaw data from LLM:")
            print("-" * 40)
            print(data)
            print("-" * 40)
            
            # Update with extracted data if available
            if not data:
                print("Warning: No data provided to _convert_to_ir")
                return ir
                
            # Update paper metadata
            if "paper_metadata" in data and data["paper_metadata"]:
                meta = data["paper_metadata"]
                if "title" in meta:
                    ir.paper_metadata.title = meta["title"] or ""
                if "authors" in meta and meta["authors"]:
                    ir.paper_metadata.authors = meta["authors"]
                if "year" in meta and meta["year"]:
                    ir.paper_metadata.year = meta["year"]
                if "doi" in meta:
                    ir.paper_metadata.doi = meta["doi"] or ""
                if "url" in meta:
                    ir.paper_metadata.url = meta["url"] or ""
                
            # Update model architecture
            if "model" in data and data["model"]:
                model_data = data["model"]
                if "name" in model_data:
                    ir.model.name = model_data["name"] or ""
                if "type" in model_data:
                    try:
                        model_type = model_data["type"]
                        if model_type is None:
                            print("Warning: model type is None, defaulting to 'nlp'")
                            ir.model.type = "nlp"
                        else:
                            ir.model.type = str(model_type).lower()
                    except Exception as e:
                        print(f"Warning: Error processing model type: {e}, defaulting to 'nlp'")
                        ir.model.type = "nlp"
                
                # Add layers
                if "layers" in model_data and isinstance(model_data["layers"], list):
                    ir.model.layers = []
                    for i, layer_data in enumerate(model_data["layers"]):
                        if not isinstance(layer_data, dict):
                            print(f"Warning: Layer {i} is not a dictionary, skipping")
                            continue
                        try:
                            layer_type = str(layer_data.get("type", "")).strip()
                            if not layer_type:
                                print(f"Warning: Layer {i} has no type, skipping")
                                continue
                                
                            layer_type = self._map_layer_type(layer_type)
                            layer = LayerSpec(
                                name=str(layer_data.get("name", f"layer_{i}")),
                                type=layer_type,
                                parameters=layer_data.get("parameters", {}) or {}
                            )
                            ir.model.layers.append(layer)
                        except Exception as e:
                            print(f"Warning: Could not process layer {i}: {e}")
                            continue
            
            # Update training config
            if "training" in data and data["training"]:
                train_data = data["training"]
                
                # Handle dataset
                if "dataset" in train_data and isinstance(train_data["dataset"], dict):
                    ds_data = train_data["dataset"]
                    ir.training.dataset = DatasetSpec(
                        name=str(ds_data.get("name", "")),
                        input_shape=ds_data.get("input_shape") or ["batch_size", "seq_len"],
                        output_shape=ds_data.get("output_shape") or ["batch_size", "num_classes"],
                        preprocessing=ds_data.get("preprocessing") or {},
                        source=ds_data.get("source")
                    )
                
                # Handle other training parameters
                if "batch_size" in train_data and train_data["batch_size"] is not None:
                    ir.training.batch_size = int(train_data["batch_size"])
                    
                if "epochs" in train_data and train_data["epochs"] is not None:
                    ir.training.epochs = int(train_data["epochs"])
                    
                if "optimizer" in train_data and isinstance(train_data["optimizer"], dict):
                    opt_data = train_data["optimizer"]
                    opt_type = str(opt_data.get("type", "adam")).lower() or "adam"  # Default to 'adam' if empty or None
                    
                    # Handle case where opt_type might be 'none' or invalid
                    valid_optimizers = ["adam", "adamw", "sgd", "rmsprop", "adagrad", "adadelta", "adamax", "nadam"]
                    if opt_type not in valid_optimizers:
                        print(f"Warning: Invalid optimizer type '{opt_type}', defaulting to 'adam'")
                        opt_type = "adam"
                        
                    ir.training.optimizer = OptimizerSpec(
                        type=opt_type,
                        parameters=opt_data.get("parameters") or {}
                    )
                    
                if "loss" in train_data and train_data["loss"] is not None:
                    ir.training.loss = str(train_data["loss"])
                    
                if "metrics" in train_data and isinstance(train_data["metrics"], list):
                    ir.training.metrics = [str(m) for m in train_data["metrics"] if m]
                    
                if "learning_rate" in train_data and train_data["learning_rate"] is not None:
                    ir.training.learning_rate = float(train_data["learning_rate"])
            
            # Update evaluation metrics
            if "evaluation" in data and data["evaluation"]:
                eval_data = data["evaluation"]
                if "metrics" in eval_data and isinstance(eval_data["metrics"], dict):
                    ir.evaluation.metrics = {str(k): float(v) for k, v in eval_data["metrics"].items() 
                                          if v is not None}
                    
                if "results" in eval_data and isinstance(eval_data["results"], dict):
                    ir.evaluation.results = {str(k): float(v) for k, v in eval_data["results"].items()
                                          if v is not None}
            
            return ir
            
        except Exception as e:
            print(f"\nError in _convert_to_ir: {str(e)}\n")
            print(f"Data being processed: {data}\n")
            import traceback
            traceback.print_exc()
            return create_default_ir()