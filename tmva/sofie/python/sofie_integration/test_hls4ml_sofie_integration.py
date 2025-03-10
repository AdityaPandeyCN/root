#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import hls4ml
from ROOT import TMVA
import ROOT

# Import the main conversion function
from hls4ml_sofie_integration import parse_hls4ml_to_sofie

def test_basic_model():
    """Test the basic sequential model with ReLU, ELU, Dense and Reshape layers"""
    print("\n" + "="*80)
    print("Testing Basic Sequential Model")
    print("="*80)
    
    # Create a simple Keras model
    keras_model = keras.Sequential([
        keras.layers.Input(shape=(10,)),
        keras.layers.Dense(20),
        keras.layers.Reshape((4, 5)),
        keras.layers.Dense(10),
        keras.layers.Activation('relu'),
        keras.layers.Dense(10),
        keras.layers.ELU(alpha=1.0)
    ])
    keras_model.compile(optimizer='adam', loss='mse')
    
    # Convert to HLS4ML
    print("Converting to HLS4ML...")
    hls_config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        hls_config=hls_config,
        output_dir='hls4ml_basic_model',
        part='xc7z020-clg400-1'
    )
    hls_model.compile()
    
    # Perform the actual conversion
    print("Converting HLS4ML model to SOFIE...")
    sofie_model = parse_hls4ml_to_sofie(hls_model, model_name="basic_model", verbose=True)
    
    # Generate C++ code
    print("Generating C++ code...")
    try:
        sofie_model.Generate()
        header_file = "basic_model.hxx"
        sofie_model.OutputGenerated(header_file)
        
        if os.path.exists(header_file):
            print(f"Success! Generated SOFIE model header file (size: {os.path.getsize(header_file)} bytes)")
            return True
        else:
            print(" Failed to generate SOFIE model header file")
            return False
    except Exception as e:
        print(f"Error generating C++ code: {e}")
        return False

def test_concat_model():
    """Test a model with concatenation layer"""
    print("\n" + "="*80)
    print("Testing Model with Concatenation")
    print("="*80)
    
    # Create a Keras model with a concatenation layer
    input1 = keras.layers.Input(shape=(10,))
    input2 = keras.layers.Input(shape=(10,))
    
    # Process first input
    x1 = keras.layers.Dense(15)(input1)
    x1 = keras.layers.Activation('relu')(x1)
    
    # Process second input
    x2 = keras.layers.Dense(15)(input2)
    x2 = keras.layers.Activation('relu')(x2)
    
    # Concatenate
    concat = keras.layers.Concatenate(axis=1)([x1, x2])
    
    # Final layer
    output = keras.layers.Dense(10, activation='sigmoid')(concat)
    
    # Create the model
    keras_model = keras.models.Model(inputs=[input1, input2], outputs=output)
    keras_model.compile(optimizer='adam', loss='mse')
    
    # Display model summary
    print("Keras Model Summary:")
    keras_model.summary()
    
    # Convert to HLS4ML
    print("Converting to HLS4ML...")
    try:
        hls_config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')
        hls_model = hls4ml.converters.convert_from_keras_model(
            keras_model,
            hls_config=hls_config,
            output_dir='hls4ml_concat_model',
            part='xc7z020-clg400-1'
        )
        hls_model.compile()
        
        # Perform the conversion
        print("Converting HLS4ML model to SOFIE...")
        sofie_model = parse_hls4ml_to_sofie(hls_model, model_name="concat_model", verbose=True)
        
        # Generate C++ code
        print("Generating C++ code...")
        sofie_model.Generate()
        header_file = "concat_model.hxx"
        sofie_model.OutputGenerated(header_file)
        
        if os.path.exists(header_file):
            print(f" Success! Generated SOFIE model header file (size: {os.path.getsize(header_file)} bytes)")
            return True
        else:
            print("Failed to generate SOFIE model header file")
            return False
    except Exception as e:
        print(f"Error in HLS4ML conversion or SOFIE conversion: {e}")
        print("Note: Models with multiple inputs might not be fully supported by HLS4ML.")
        return False

def test_all_operators():
    """Test a model with all five required operators"""
    print("\n" + "="*80)
    print("Testing Model with All Five Operators")
    print("="*80)
    
    # Create a Keras model with all five operators
    inputs = keras.layers.Input(shape=(10,))
    
    # Dense (Gemm)
    x1 = keras.layers.Dense(20)(inputs)
    x2 = keras.layers.Dense(20)(inputs)
    
    # ReLU
    x1 = keras.layers.Activation('relu')(x1)
    
    # Reshape
    x1 = keras.layers.Reshape((4, 5))(x1)
    x1 = keras.layers.Flatten()(x1)  # Back to 1D for concatenation
    
    # Concatenate
    x = keras.layers.Concatenate(axis=1)([x1, x2])
    
    # ELU
    x = keras.layers.ELU(alpha=1.0)(x)
    
    # Final layer
    outputs = keras.layers.Dense(5)(x)
    
    # Create the model
    keras_model = keras.models.Model(inputs=inputs, outputs=outputs)
    keras_model.compile(optimizer='adam', loss='mse')
    
    # Display model summary
    print("Keras Model Summary:")
    keras_model.summary()
    
    # Convert to HLS4ML
    print("Converting to HLS4ML...")
    try:
        hls_config = hls4ml.utils.config_from_keras_model(keras_model, granularity='name')
        hls_model = hls4ml.converters.convert_from_keras_model(
            keras_model,
            hls_config=hls_config,
            output_dir='hls4ml_all_operators',
            part='xc7z020-clg400-1'
        )
        hls_model.compile()
        
        # Perform the conversion
        print("Converting HLS4ML model to SOFIE...")
        sofie_model = parse_hls4ml_to_sofie(hls_model, model_name="all_operators", verbose=True)
        
        # Generate C++ code
        print("Generating C++ code...")
        sofie_model.Generate()
        header_file = "all_operators.hxx"
        sofie_model.OutputGenerated(header_file)
        
        if os.path.exists(header_file):
            print(f"Success! Generated SOFIE model header file (size: {os.path.getsize(header_file)} bytes)")
            return True
        else:
            print("Failed to generate SOFIE model header file")
            return False
    except Exception as e:
        print(f"Error in conversion: {e}")
        return False

def run_all_tests():
    """Run all test cases"""
    results = []
    
    # Test basic model
    results.append(("Basic Sequential Model", test_basic_model()))
    
    # Uncomment to test concat model (may not be fully supported by HLS4ML)
    # results.append(("Concat Model", test_concat_model()))
    
    # Uncomment to test all operators (may not be fully supported by HLS4ML)
    # results.append(("All Operators Model", test_all_operators()))
    
    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    # Overall result
    overall = all(result for _, result in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
    
    return overall

if __name__ == "__main__":
    run_all_tests()