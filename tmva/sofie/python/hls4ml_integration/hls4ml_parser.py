#!/usr/bin/env python3
import numpy as np
import json
from typing import Dict, Any, Optional, Union, List
import logging

logger = logging.getLogger("ONNX_HLS4ML_Parser")

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle NumPy array and numeric types.
    
    Converts NumPy arrays to lists and NumPy numeric types to Python native types.
    """
    def default(self, obj: Union[np.ndarray, np.number]) -> Union[List, int, float]:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)

def parse_hls4ml_model(hls_model: Any, verbose: bool = True) -> Optional[Dict[str, Any]]:
    """
    Parse an HLS4ML model and extract comprehensive configuration and performance metrics.

    Args:
        hls_model: The HLS4ML model to be parsed
        verbose: Flag to control logging verbosity (default: True)

    Returns:
        A dictionary containing model metadata, layer configurations, 
        model structure, and performance metrics, or None if parsing fails
    """
    if hls_model is None:
        logger.error("Input model is None")
        return None

    try:
        config: Dict[str, Any] = {
            "model_metadata": {
                "name": hls_model.config.config.get("ProjectName", "Unnamed Project"),
                "reuse_factor": hls_model.config.config.get("ReuseFactor", 1),
                "clock_period": hls_model.config.config.get("ClockPeriod", 5),
                "backend": hls_model.config.config.get("Backend", "Vivado")
            },
            "layers": {},
            "model_structure": {
                "input_layers": [],
                "output_layers": [],
                "layer_sequence": [],
                "total_layers": 0
            },
            "performance_metrics": {
                "total_parameters": 0,
                "estimated_macs": 0
            }
        }

        for layer_name, layer in hls_model.graph.items():
            raw_type = layer.__class__.__name__
            layer_type = raw_type.replace("Vivado", "").replace("Resource", "")
            
            layer_config: Dict[str, Any] = {
                "type": layer_type,
                "base_attributes": {},
                "type_attributes": {},
                "inputs": [],
                "outputs": [],
                "weights": {},
                "computational_metrics": {
                    "parameters": 0,
                    "macs": 0
                },
                "input_shape": None,
                "output_shape": None
            }

            try:
                # Input shape extraction
                try:
                    input_var = layer.get_input_variable() if hasattr(layer, 'get_input_variable') else None
                    layer_config["input_shape"] = input_var.shape if input_var else None
                except Exception as e:
                    logger.debug(f"Input shape error for {layer_name}: {str(e)}")

                # Output shape extraction
                try:
                    output_var = layer.get_output_variable() if hasattr(layer, 'get_output_variable') else None
                    layer_config["output_shape"] = output_var.shape if output_var else None
                except Exception as e:
                    logger.debug(f"Output shape error for {layer_name}: {str(e)}")

                # Convolutional layer parameter extraction
                if "Conv" in raw_type:
                    _process_conv_layer(layer, layer_name, layer_config)

                # Input/Output layer detection
                if "Input" in raw_type:
                    config["model_structure"]["input_layers"].append(layer_name)
                if layer_name == list(hls_model.graph.keys())[-1]:
                    config["model_structure"]["output_layers"].append(layer_name)

            except Exception as e:
                logger.warning(f"Error processing {layer_name}: {str(e)}")

            config["model_structure"]["layer_sequence"].append(layer_name)
            config["model_structure"]["total_layers"] += 1
            config["performance_metrics"]["total_parameters"] += layer_config["computational_metrics"]["parameters"]
            config["performance_metrics"]["estimated_macs"] += layer_config["computational_metrics"]["macs"]
            
            config["layers"][layer_name] = layer_config

        logger.info(f"Successfully parsed {config['model_structure']['total_layers']} layers")
        logger.info(f"Total parameters: {config['performance_metrics']['total_parameters']}")
        logger.info(f"Estimated MACs: {config['performance_metrics']['estimated_macs']}")

        return config

    except Exception as e:
        logger.error(f"Model parsing failed: {str(e)}", exc_info=True)
        return None

def _process_conv_layer(layer: Any, layer_name: str, layer_config: Dict[str, Any]) -> None:
    """
    Process convolutional layer to extract weights and compute computational metrics.

    Args:
        layer: The convolutional layer to process
        layer_name: Name of the layer
        layer_config: Configuration dictionary to be updated with layer details
    """
    params = 0
    
    if hasattr(layer, 'get_weights'):
        weights_data = layer.get_weights()
        weights_list = _extract_weights(weights_data, layer_name)
        
        # Kernel weights extraction
        if len(weights_list) > 0:
            kernel_data = _sanitize_weight_data(weights_list[0], layer_name)
            if isinstance(kernel_data, np.ndarray):
                kernel_size = int(np.prod(kernel_data.shape))
                layer_config["weights"]["kernel"] = {
                    "shape": kernel_data.shape,
                    "size": kernel_size,
                    "dtype": str(kernel_data.dtype)
                }
                params += kernel_size
        
        # Bias weights extraction
        if len(weights_list) > 1:
            bias_data = _sanitize_weight_data(weights_list[1], layer_name)
            if isinstance(bias_data, np.ndarray):
                bias_size = int(np.prod(bias_data.shape))
                layer_config["weights"]["bias"] = {
                    "shape": bias_data.shape,
                    "size": bias_size,
                    "dtype": str(bias_data.dtype)
                }
                params += bias_size
    
    layer_config["computational_metrics"]["parameters"] = params
    _compute_layer_macs(layer_config)

def _extract_weights(weights_data: Any, layer_name: str) -> List[Any]:
    """
    Extract weights from various input types.

    Args:
        weights_data: Raw weights data
        layer_name: Name of the layer for logging

    Returns:
        List of extracted weights
    """
    if isinstance(weights_data, dict):
        return list(weights_data.values())
    
    try:
        return list(weights_data)
    except Exception as e:
        logger.error(f"Error converting weights for {layer_name}: {e}")
        return []

def _sanitize_weight_data(weight_data: Any, layer_name: str) -> Optional[np.ndarray]:
    """
    Sanitize weight data to ensure it's a NumPy array.

    Args:
        weight_data: Raw weight data
        layer_name: Name of the layer for logging

    Returns:
        NumPy array or None
    """
    if not isinstance(weight_data, np.ndarray):
        for attr in ['data', 'value']:
            if hasattr(weight_data, attr):
                weight_data = getattr(weight_data, attr)
                break
    
    return weight_data if isinstance(weight_data, np.ndarray) else None

def _compute_layer_macs(layer_config: Dict[str, Any]) -> None:
    """
    Compute Multiply-Accumulate (MAC) operations for a layer.

    Args:
        layer_config: Layer configuration dictionary to update
    """
    try:
        if "kernel" not in layer_config["weights"]:
            return

        kernel_shape = layer_config["weights"]["kernel"]["shape"]
        if len(kernel_shape) != 4:
            return

        kernel_height, kernel_width, in_channels, out_channels = kernel_shape
        output_shape = layer_config["output_shape"]

        if not output_shape:
            return

        out_height, out_width, _ = output_shape
        macs = out_height * out_width * kernel_height * kernel_width * in_channels * out_channels
        layer_config["computational_metrics"]["macs"] = int(macs)
    except Exception as e:
        logger.warning(f"MACs calculation error: {str(e)}")

if __name__ == '__main__':
    print("hls4ml_parser module loaded.")