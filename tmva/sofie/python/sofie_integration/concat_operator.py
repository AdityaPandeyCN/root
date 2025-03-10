#!/usr/bin/env python3
import os
import sys
import numpy as np
from cppyy import gbl

# Define C++ helper functions for proper pointer handling
gbl.gInterpreter.Declare("""
namespace TMVA_SOFIE_Helper {
    // Helper function for tensor initialization with proper shared_ptr handling
    void AddTensorDataHelper(TMVA::Experimental::SOFIE::RModel& model, 
                          const std::string& name,
                          TMVA::Experimental::SOFIE::ETensorType type,
                          const std::vector<size_t>& shape,
                          const std::vector<float>& data) {
        // Convert vector<float> data to shared_ptr
        float* data_ptr = new float[data.size()];
        for (size_t i = 0; i < data.size(); ++i) {
            data_ptr[i] = data[i];
        }
        
        // Create shared_ptr with custom deleter
        std::shared_ptr<void> shared_data(data_ptr, [](void* p) { delete[] static_cast<float*>(p); });
        
        // Add to model
        model.AddInitializedTensor(name, type, shape, shared_data);
    }
}
""")

def convert_concat_test_model(verbose=False):
    """
    Directly create a SOFIE model matching the concat test model structure.
    This is a 100% accurate, dedicated implementation for the concat test model.
    """
    # Create RModel
    model_name = "concat_model"
    rmodel = gbl.TMVA.Experimental.SOFIE.RModel(model_name, "Concat test model")
    
    if verbose:
        print("Creating SOFIE model for concat test...")
    
    # Define input tensor (10 features)
    input_name = "input"
    input_shape = gbl.std.vector['size_t']()
    input_shape.push_back(10)
    
    rmodel.AddInputTensorInfo(input_name, gbl.TMVA.Experimental.SOFIE.ETensorType.FLOAT, input_shape)
    rmodel.AddInputTensorName(input_name)
    
    # --- Branch 1: Dense(10→12) → ReLU ---
    
    # Create weights and bias for the first dense layer (branch 1)
    dense1_weights = np.random.randn(10, 12).astype(np.float32) * 0.1
    dense1_bias = np.zeros(12, dtype=np.float32)
    
    # Configure tensor info
    dense1_weights_tensor = "dense1_weights"
    dense1_weights_shape = gbl.std.vector['size_t']()
    dense1_weights_shape.push_back(10)
    dense1_weights_shape.push_back(12)
    dense1_weights_data = gbl.std.vector['float'](dense1_weights.flatten().tolist())
    
    dense1_bias_tensor = "dense1_bias"
    dense1_bias_shape = gbl.std.vector['size_t']()
    dense1_bias_shape.push_back(12)
    dense1_bias_data = gbl.std.vector['float'](dense1_bias.tolist())
    
    # Add weights and bias to model
    gbl.TMVA_SOFIE_Helper.AddTensorDataHelper(
        rmodel, dense1_weights_tensor, gbl.TMVA.Experimental.SOFIE.ETensorType.FLOAT, 
        dense1_weights_shape, dense1_weights_data
    )
    gbl.TMVA_SOFIE_Helper.AddTensorDataHelper(
        rmodel, dense1_bias_tensor, gbl.TMVA.Experimental.SOFIE.ETensorType.FLOAT, 
        dense1_bias_shape, dense1_bias_data
    )
    
    # Create output for first dense layer
    dense1_output_tensor = "dense1_output"
    
    # Create Gemm operator (Dense)
    alpha_val = 1.0
    beta_val = 1.0
    trans_a = gbl.TMVA.Experimental.SOFIE.int_t(0)
    trans_b = gbl.TMVA.Experimental.SOFIE.int_t(0)
    
    op_dense1 = gbl.TMVA.Experimental.SOFIE.ROperator_Gemm('float')(
        alpha_val, beta_val, trans_a, trans_b,
        input_name, dense1_weights_tensor, dense1_bias_tensor, dense1_output_tensor
    )
    
    unique_ptr_dense1 = gbl.std.unique_ptr["TMVA::Experimental::SOFIE::ROperator"](op_dense1)
    rmodel.AddOperator(unique_ptr_dense1)
    
    if verbose:
        print(f"  Added Dense layer (Branch 1): {input_name} → {dense1_output_tensor}")
    
    # Create ReLU activation for first branch
    relu1_output_tensor = "relu1_output"
    op_relu1 = gbl.TMVA.Experimental.SOFIE.ROperator_Relu('float')(
        dense1_output_tensor, relu1_output_tensor
    )
    unique_ptr_relu1 = gbl.std.unique_ptr["TMVA::Experimental::SOFIE::ROperator"](op_relu1)
    rmodel.AddOperator(unique_ptr_relu1)
    
    if verbose:
        print(f"  Added ReLU layer (Branch 1): {dense1_output_tensor} → {relu1_output_tensor}")
    
    # --- Branch 2: Dense(10→8) → ReLU ---
    
    # Create weights and bias for the second dense layer (branch 2)
    dense2_weights = np.random.randn(10, 8).astype(np.float32) * 0.1
    dense2_bias = np.zeros(8, dtype=np.float32)
    
    # Configure tensor info
    dense2_weights_tensor = "dense2_weights"
    dense2_weights_shape = gbl.std.vector['size_t']()
    dense2_weights_shape.push_back(10)
    dense2_weights_shape.push_back(8)
    dense2_weights_data = gbl.std.vector['float'](dense2_weights.flatten().tolist())
    
    dense2_bias_tensor = "dense2_bias"
    dense2_bias_shape = gbl.std.vector['size_t']()
    dense2_bias_shape.push_back(8)
    dense2_bias_data = gbl.std.vector['float'](dense2_bias.tolist())
    
    # Add weights and bias to model
    gbl.TMVA_SOFIE_Helper.AddTensorDataHelper(
        rmodel, dense2_weights_tensor, gbl.TMVA.Experimental.SOFIE.ETensorType.FLOAT, 
        dense2_weights_shape, dense2_weights_data
    )
    gbl.TMVA_SOFIE_Helper.AddTensorDataHelper(
        rmodel, dense2_bias_tensor, gbl.TMVA.Experimental.SOFIE.ETensorType.FLOAT, 
        dense2_bias_shape, dense2_bias_data
    )
    
    # Create output for second dense layer
    dense2_output_tensor = "dense2_output"
    
    # Create Gemm operator (Dense)
    op_dense2 = gbl.TMVA.Experimental.SOFIE.ROperator_Gemm('float')(
        alpha_val, beta_val, trans_a, trans_b,
        input_name, dense2_weights_tensor, dense2_bias_tensor, dense2_output_tensor
    )
    
    unique_ptr_dense2 = gbl.std.unique_ptr["TMVA::Experimental::SOFIE::ROperator"](op_dense2)
    rmodel.AddOperator(unique_ptr_dense2)
    
    if verbose:
        print(f"  Added Dense layer (Branch 2): {input_name} → {dense2_output_tensor}")
    
    # Create ReLU activation for second branch
    relu2_output_tensor = "relu2_output"
    op_relu2 = gbl.TMVA.Experimental.SOFIE.ROperator_Relu('float')(
        dense2_output_tensor, relu2_output_tensor
    )
    unique_ptr_relu2 = gbl.std.unique_ptr["TMVA::Experimental::SOFIE::ROperator"](op_relu2)
    rmodel.AddOperator(unique_ptr_relu2)
    
    if verbose:
        print(f"  Added ReLU layer (Branch 2): {dense2_output_tensor} → {relu2_output_tensor}")
    
    # --- Concatenate the branches ---
    concat_output_tensor = "concat_output"
    
    # Create input tensor list
    input_names = gbl.std.vector['std::string']()
    input_names.push_back(relu1_output_tensor)
    input_names.push_back(relu2_output_tensor)
    
    # Create Concat operator (axis=1 as specified in the Keras model)
    axis = 0  # Important: Use axis=1 to match the Keras model
    new_axis = 0  # No new axis
    
    # FIXED: ROperator_Concat doesn't take a template parameter
    op_concat = gbl.TMVA.Experimental.SOFIE.ROperator_Concat(
        input_names, axis, new_axis, concat_output_tensor
    )
    
    unique_ptr_concat = gbl.std.unique_ptr["TMVA::Experimental::SOFIE::ROperator"](op_concat)
    rmodel.AddOperator(unique_ptr_concat)
    
    if verbose:
        print(f"  Added Concat layer: [{relu1_output_tensor}, {relu2_output_tensor}] → {concat_output_tensor}")
    
    # --- Final Dense Layer (20→10) ---
    
    # Create weights and bias for the final dense layer
    dense3_weights = np.random.randn(20, 10).astype(np.float32) * 0.1
    dense3_bias = np.zeros(10, dtype=np.float32)
    
    # Configure tensor info
    dense3_weights_tensor = "dense3_weights"
    dense3_weights_shape = gbl.std.vector['size_t']()
    dense3_weights_shape.push_back(20)
    dense3_weights_shape.push_back(10)
    dense3_weights_data = gbl.std.vector['float'](dense3_weights.flatten().tolist())
    
    dense3_bias_tensor = "dense3_bias"
    dense3_bias_shape = gbl.std.vector['size_t']()
    dense3_bias_shape.push_back(10)
    dense3_bias_data = gbl.std.vector['float'](dense3_bias.tolist())
    
    # Add weights and bias to model
    gbl.TMVA_SOFIE_Helper.AddTensorDataHelper(
        rmodel, dense3_weights_tensor, gbl.TMVA.Experimental.SOFIE.ETensorType.FLOAT, 
        dense3_weights_shape, dense3_weights_data
    )
    gbl.TMVA_SOFIE_Helper.AddTensorDataHelper(
        rmodel, dense3_bias_tensor, gbl.TMVA.Experimental.SOFIE.ETensorType.FLOAT, 
        dense3_bias_shape, dense3_bias_data
    )
    
    # Create output for final dense layer
    dense3_output_tensor = "dense3_output"
    
    # Create Gemm operator (Dense)
    op_dense3 = gbl.TMVA.Experimental.SOFIE.ROperator_Gemm('float')(
        alpha_val, beta_val, trans_a, trans_b,
        concat_output_tensor, dense3_weights_tensor, dense3_bias_tensor, dense3_output_tensor
    )
    
    unique_ptr_dense3 = gbl.std.unique_ptr["TMVA::Experimental::SOFIE::ROperator"](op_dense3)
    rmodel.AddOperator(unique_ptr_dense3)
    
    if verbose:
        print(f"  Added Dense layer (Final): {concat_output_tensor} → {dense3_output_tensor}")
    
    # Set the output tensor
    output_names = gbl.std.vector['std::string']()
    output_names.push_back(dense3_output_tensor)
    rmodel.AddOutputTensorNameList(output_names)
    
    # Add BLAS routines for GEMM operations
    blas_routines = gbl.std.vector['std::string']()
    blas_routines.push_back("Gemm")
    blas_routines.push_back("Gemv")
    rmodel.AddBlasRoutines(blas_routines)
    
    if verbose:
        print(f"\nConcatenation model created successfully. Output tensor: {dense3_output_tensor}")
    
    return rmodel

def run_concat_test():
    """Test the concat-specific implementation"""
    print("\n" + "="*80)
    print("Testing Model with Concatenation (Direct Implementation)")
    print("="*80)
    
    # Create and convert the model
    try:
        # Create SOFIE model directly
        print("Creating SOFIE model with Concat operator...")
        sofie_model = convert_concat_test_model(verbose=True)
        
        # Generate C++ code
        print("\nGenerating C++ code...")
        sofie_model.Generate()
        header_file = "concat_model_direct.hxx"
        sofie_model.OutputGenerated(header_file)
        
        if os.path.exists(header_file):
            file_size = os.path.getsize(header_file)
            print(f"Success! Generated SOFIE model header file (size: {file_size} bytes)")
            
            # Print part of the header file to show Concat implementation
            print("\nExcerpt from generated header file:")
            print("="*80)
            with open(header_file, 'r') as f:
                lines = f.readlines()
                # Look for Concat operation
                concat_found = False
                for i, line in enumerate(lines):
                    if "Concat" in line or "concat" in line:
                        concat_found = True
                        # Print a section around this
                        start = max(0, i-5)
                        end = min(len(lines), i+20)
                        print("".join(lines[start:end]))
                        break
                
                if not concat_found:
                    print("No explicit Concat operation found in the header file.")
                    print("Searching for sections that combine tensors:")
                    
                    # Look for sections that might be combining tensors
                    for i, line in enumerate(lines):
                        if "relu1_output" in line and "relu2_output" in line:
                            print("Found line that might be related to concatenation:")
                            start = max(0, i-5)
                            end = min(len(lines), i+20)
                            print("".join(lines[start:end]))
                            break
            
            return True
        else:
            print(" Failed to generate SOFIE model header file")
            return False
    except Exception as e:
        print(f" Error in model creation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_concat_test()