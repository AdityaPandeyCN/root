#!/usr/bin/env python3
"""
Test script for hls4ml_parser module.
Creates a sample neural network, converts it to HLS4ML format,
and tests the parser's ability to extract model configuration.
"""

import os
import sys
import json
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HLS4ML_Parser_Test")

def create_test_model():
    """Create a simple test Keras model."""
    try:
        import tensorflow as tf
        
        logger.info("Creating sample Keras model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(5, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy')
        logger.info("Model created successfully")
        return model
    except ImportError:
        logger.error("TensorFlow not installed. Please install tensorflow package.")
        return None

def convert_to_hls4ml(model, output_dir='hls4ml_test_model', part='xcu250-figd2104-2L-e'):
    """Convert Keras model to HLS4ML model."""
    try:
        import hls4ml
        
        logger.info("Converting Keras model to HLS4ML...")
        hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, 
            hls_config=hls_config,
            output_dir=output_dir,
            part=part
        )
        logger.info("Conversion to HLS4ML model successful")
        return hls_model
    except ImportError:
        logger.error("HLS4ML not installed. Please install hls4ml package.")
        return None

def parse_and_save_model(hls_model, output_file='model_config.json'):
    """Parse HLS4ML model and save configuration to JSON file."""
    try:
        from hls4ml_parser import parse_hls4ml_model, NumpyEncoder
        
        logger.info("Parsing HLS4ML model...")
        config_data = parse_hls4ml_model(hls_model)
        
        if config_data:
            logger.info("Successfully parsed the model")
            
            # Save the config to a file
            with open(output_file, 'w') as f:
                json.dump(config_data, f, cls=NumpyEncoder, indent=2)
            logger.info(f"Saved model configuration to {output_file}")
            return True
        else:
            logger.error("Failed to parse the model")
            return False
    except ImportError:
        logger.error("hls4ml_parser module not found. Make sure it's in the current directory.")
        return False
    except Exception as e:
        logger.error(f"Error parsing model: {str(e)}")
        return False

def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(description='Test the HLS4ML parser module')
    parser.add_argument('--output-dir', type=str, default='hls4ml_test_model',
                        help='Output directory for HLS4ML model')
    parser.add_argument('--config-file', type=str, default='model_config.json',
                        help='Output JSON file for model configuration')
    parser.add_argument('--part', type=str, default='xcu250-figd2104-2L-e',
                        help='FPGA part for HLS4ML synthesis')
    args = parser.parse_args()
    
    # Create model
    model = create_test_model()
    if not model:
        return 1
    
    # Convert model to HLS4ML
    hls_model = convert_to_hls4ml(model, args.output_dir, args.part)
    if not hls_model:
        return 1
    
    # Parse model and save configuration
    success = parse_and_save_model(hls_model, args.config_file)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())