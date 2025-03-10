#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 10
#define N_LAYER_2 20
#define N_SIZE_0_4 4
#define N_SIZE_1_4 5
#define N_OUTPUTS_13 4
#define N_FILT_13 10
#define N_LAYER_1_5 4
#define N_LAYER_2_5 10
#define N_OUTPUTS_14 4
#define N_FILT_14 10
#define N_LAYER_1_8 4
#define N_LAYER_2_8 10


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<37,17> dense_result_t;
typedef ap_fixed<16,6> dense_weight_t;
typedef ap_fixed<16,6> dense_bias_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<57,27> dense_1_result_t;
typedef ap_fixed<16,6> dense_1_weight_t;
typedef ap_fixed<16,6> dense_1_bias_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<18,8> activation_table_t;
typedef ap_fixed<37,17> dense_2_result_t;
typedef ap_fixed<16,6> dense_2_weight_t;
typedef ap_fixed<16,6> dense_2_bias_t;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<16,6> elu_param_t;
typedef ap_fixed<18,8> elu_table_t;


#endif
