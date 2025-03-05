#!/usr/bin/env python3
import os
import json
import logging
from typing import Dict, Any, Optional
from pprint import pprint

import numpy as np
import onnx
import hls4ml
from onnx import helper

# QONNX imports
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup_model
from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
from qonnx.transformation.gemm_to_matmul import GemmToMatMul

from hls4ml_parser import parse_hls4ml_model, NumpyEncoder

def setup_logging() -> logging.Logger:
    """
    Configure and set up logging with console and file handlers.

    Returns:
        Configured logger instance
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')
    logger = logging.getLogger("ONNX_HLS4ML_Parser")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler('onnx_hls4ml_parsing.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d: %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def patch_onnx_model_attributes(onnx_model: onnx.ModelProto, logger: logging.Logger) -> onnx.ModelProto:
    """
    Add default attributes to Conv nodes in the ONNX model.

    Args:
        onnx_model: Input ONNX model
        logger: Logging instance

    Returns:
        Modified ONNX model with default attributes
    """
    # Add dilations attribute
    for node in onnx_model.graph.node:
        if node.op_type == "Conv":
            if not any(attr.name == "dilations" for attr in node.attribute):
                dilations_attr = helper.make_attribute("dilations", [1, 1])
                node.attribute.append(dilations_attr)
                logger.info(f"Added dilations=[1, 1] to Conv node: {node.name}")

    # Add group attribute
    for node in onnx_model.graph.node:
        if node.op_type == "Conv":
            if not any(attr.name == "group" for attr in node.attribute):
                group_attr = helper.make_attribute("group", 1)
                node.attribute.append(group_attr)
                logger.info(f"Added group=1 to Conv node: {node.name}")

    return onnx_model

def apply_model_transformations(model: ModelWrapper, logger: logging.Logger) -> ModelWrapper:
    """
    Apply a series of transformations to the QONNX model.

    Args:
        model: QONNX ModelWrapper instance
        logger: Logging instance

    Returns:
        Transformed model
    """
    transformation_steps = [
        ('cleanup_model', cleanup_model),
        ('ConvertToChannelsLastAndClean', lambda m: m.transform(ConvertToChannelsLastAndClean())),
        ('GemmToMatMul', lambda m: m.transform(GemmToMatMul())),
        ('final_cleanup', cleanup_model)
    ]
    
    for step_name, transform_func in transformation_steps:
        try:
            model = transform_func(model)
            logger.info(f"Completed transformation: {step_name}")
        except Exception as err:
            logger.exception(f"Transformation {step_name} failed")
            raise

    return model

def parse_onnx_model(onnx_model_path: str, output_dir: str = 'hls4ml_prj_onnx') -> Optional[Dict[str, Any]]:
    """
    Comprehensive ONNX model parsing with preprocessing and error handling.

    Args:
        onnx_model_path: Path to the ONNX model file
        output_dir: Output directory for the HLS4ML project

    Returns:
        Parsed model configuration or None if parsing fails
    """
    logger = setup_logging()

    try:
        # Load and validate ONNX model
        original_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(original_model)
        logger.info("Original ONNX model loaded and validated")
    except Exception as load_error:
        logger.exception("Failed to load and validate the ONNX model")
        return None

    # Patch ONNX model attributes
    try:
        patched_model = patch_onnx_model_attributes(original_model, logger)
        patched_model_path = os.path.splitext(onnx_model_path)[0] + "_patched.onnx"
        onnx.save(patched_model, patched_model_path)
        logger.info(f"Patched ONNX model saved as {patched_model_path}")
    except Exception as patch_error:
        logger.exception("Failed to patch the ONNX model")
        return None

    # Wrap the patched model
    try:
        model = ModelWrapper(patched_model_path)
        logger.info("Converted ONNX model to ModelWrapper")
    except Exception as wrap_error:
        logger.exception("Failed to wrap the ONNX model")
        return None

    # Apply transformations
    try:
        model = apply_model_transformations(model, logger)
    except Exception:
        return None

    # Generate HLS4ML configuration
    try:
        config = hls4ml.utils.config.config_from_onnx_model(
            model,
            granularity='name',
            backend='Vivado',
            default_precision='fixed<16,6>',
            default_reuse_factor=1
        )
        logger.info("Generated HLS4ML configuration from the ONNX model")
    except Exception as config_error:
        logger.exception("Failed to generate HLS4ML configuration")
        return None

    # Convert to HLS4ML model
    try:
        hls_model = hls4ml.converters.convert_from_onnx_model(
            model,
            output_dir=output_dir,
            io_type='io_stream',
            backend='Vivado',
            hls_config=config,
            part='xcu250-figd2104-2L-e',
            vsynth=True        
        )
        logger.info("Converted the ONNX model to an HLS4ML model instance")
    except Exception as conversion_error:
        logger.exception("Failed to convert to an HLS4ML model")
        return None

    # Parse the HLS4ML model
    try:
        model_config = parse_hls4ml_model(hls_model)
        if model_config is None:
            logger.error("Model parsing returned None")
            return None
        
        # Save parsed configuration
        output_config_path = os.path.join(output_dir, 'parsed_model_config_onnx.json')
        os.makedirs(output_dir, exist_ok=True)
        with open(output_config_path, "w") as f:
            json.dump(model_config, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Model configuration saved to {output_config_path}")
        return model_config
    except Exception as parsing_error:
        logger.exception("ONNX model parsing failed")
        return None

def test_onnx_model_parsing():
    """
    Test function for ONNX model parsing with comprehensive error handling.
    """
    onnx_model_paths = [
        "/home/aditya/Downloads/ConvWithAsymmetricPadding.onnx",
        "ConvWithAsymmetricPadding.onnx",
    ]

    # Find the first existing ONNX model file
    onnx_model_path = next((path for path in onnx_model_paths if os.path.exists(path)), None)

    if not onnx_model_path:
        print("Error: No ONNX model file found. Please provide a valid path.")
        return None

    print(f"Processing ONNX model from: {onnx_model_path}")
    
    # Run the ONNX parsing function
    model_config = parse_onnx_model(onnx_model_path)
    
    if model_config:
        print("\nONNX Model Configuration Preview:")
        preview_config = {
            "Layers": len(model_config.get('layers', {})),
            "Input Layers": model_config.get('model_structure', {}).get('input_layers', []),
            "Output Layers": model_config.get('model_structure', {}).get('output_layers', []),
            "Total Parameters": model_config.get('performance_metrics', {}).get('total_parameters', 0)
        }
        pprint(preview_config)
        
        print("\nDetailed Layer Information:")
        for layer_name, layer_info in model_config.get('layers', {}).items():
            print(f"Layer: {layer_name}")
            print(f"  Type: {layer_info.get('type', 'Unknown')}")
            print(f"  Input Shape: {layer_info.get('input_shape', 'N/A')}")
            print(f"  Output Shape: {layer_info.get('output_shape', 'N/A')}")
            print(f"  Parameters: {layer_info.get('computational_metrics', {}).get('parameters', 0)}")
            print()
    else:
        print("Model parsing failed. Check the log file for detailed error information.")
    
    return model_config

if __name__ == '__main__':
    test_onnx_model_parsing()