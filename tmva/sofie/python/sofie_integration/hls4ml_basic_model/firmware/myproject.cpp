#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t input_1[N_INPUT_1_1],
    result_t layer10_out[N_LAYER_1_8*N_LAYER_2_8]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_1,layer10_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<dense_weight_t, 200>(w2, "w2.txt");
        nnet::load_weights_from_txt<dense_bias_t, 20>(b2, "b2.txt");
        nnet::load_weights_from_txt<dense_1_weight_t, 50>(w13, "w13.txt");
        nnet::load_weights_from_txt<dense_1_bias_t, 10>(b13, "b13.txt");
        nnet::load_weights_from_txt<dense_2_weight_t, 100>(w14, "w14.txt");
        nnet::load_weights_from_txt<dense_2_bias_t, 10>(b14, "b14.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    dense_result_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::dense<input_t, dense_result_t, config2>(input_1, layer2_out, w2, b2); // dense

    auto& layer4_out = layer2_out;
    dense_1_result_t layer13_out[N_OUTPUTS_13*N_FILT_13];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::pointwise_conv_1d_cl<dense_result_t, dense_1_result_t, config13>(layer4_out, layer13_out, w13, b13); // dense_1

    layer7_t layer7_out[N_LAYER_1_5*N_LAYER_2_5];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<dense_1_result_t, layer7_t, relu_config7>(layer13_out, layer7_out); // activation

    dense_2_result_t layer14_out[N_OUTPUTS_14*N_FILT_14];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::pointwise_conv_1d_cl<layer7_t, dense_2_result_t, config14>(layer7_out, layer14_out, w14, b14); // dense_2

    nnet::elu<dense_2_result_t, elu_param_t, result_t, ELU_config10>(layer14_out, 1.0, layer10_out); // elu

}

