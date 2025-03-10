//Code generated automatically by TMVA for Inference of Model file [basic_model] at [HLS4ML conversio] 

#ifndef ROOT_TMVA_SOFIE_BASIC_MODEL
#define ROOT_TMVA_SOFIE_BASIC_MODEL

#include <algorithm>
#include <vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_basic_model{
namespace BLAS{
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
struct Session {
// initialized tensors
std::vector<float> fTensor_dense2_bias = std::vector<float>(10);
float * tensor_dense2_bias = fTensor_dense2_bias.data();
std::vector<float> fTensor_dense2_weights = std::vector<float>(100);
float * tensor_dense2_weights = fTensor_dense2_weights.data();
std::vector<float> fTensor_dense1_bias = std::vector<float>(10);
float * tensor_dense1_bias = fTensor_dense1_bias.data();
std::vector<float> fTensor_dense1_weights = std::vector<float>(50);
float * tensor_dense1_weights = fTensor_dense1_weights.data();
std::vector<float> fTensor_dense_bias = std::vector<float>(20);
float * tensor_dense_bias = fTensor_dense_bias.data();
std::vector<float> fTensor_dense_weights = std::vector<float>(200);
float * tensor_dense_weights = fTensor_dense_weights.data();

//--- declare and allocate the intermediate tensors
std::vector<float> fTensor_elu_output = std::vector<float>(40);
float * tensor_elu_output = fTensor_elu_output.data();
std::vector<float> fTensor_dense2_output = std::vector<float>(40);
float * tensor_dense2_output = fTensor_dense2_output.data();
std::vector<float> fTensor_dense2_biasbcast = std::vector<float>(40);
float * tensor_dense2_biasbcast = fTensor_dense2_biasbcast.data();
std::vector<float> fTensor_dense1_output = std::vector<float>(40);
float * tensor_dense1_output = fTensor_dense1_output.data();
std::vector<float> fTensor_dense1_biasbcast = std::vector<float>(40);
float * tensor_dense1_biasbcast = fTensor_dense1_biasbcast.data();
std::vector<float> fTensor_relu_output = std::vector<float>(40);
float * tensor_relu_output = fTensor_relu_output.data();
std::vector<float> fTensor_reshape_output = std::vector<float>(20);
float * tensor_reshape_output = fTensor_reshape_output.data();
std::vector<float> fTensor_dense_output = std::vector<float>(20);
float * tensor_dense_output = fTensor_dense_output.data();


Session(std::string filename ="basic_model.dat") {

//--- reading weights from file
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("tmva-sofie failed to open file " + filename + " for input weights");
   }
   std::string tensor_name;
   size_t length;
   f >> tensor_name >> length;
   if (tensor_name != "tensor_dense2_bias" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_dense2_bias , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 10) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 10 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_dense2_bias[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_dense2_bias");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_dense2_weights" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_dense2_weights , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 100) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 100 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_dense2_weights[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_dense2_weights");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_dense1_bias" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_dense1_bias , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 10) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 10 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_dense1_bias[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_dense1_bias");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_dense1_weights" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_dense1_weights , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 50) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 50 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_dense1_weights[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_dense1_weights");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_dense_bias" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_dense_bias , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 20) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 20 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_dense_bias[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_dense_bias");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_dense_weights" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_dense_weights , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 200) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 200 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_dense_weights[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_dense_weights");
   }
   f.close();

//---- allocate the intermediate dynamic tensors
//--- broadcast bias tensor dense1_biasfor Gemm op
   {
      float * data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_dense1_bias,{ 10 }, { 4 , 10 });
      std::copy(data, data + 40, tensor_dense1_biasbcast);
      delete [] data;
   }
//--- broadcast bias tensor dense2_biasfor Gemm op
   {
      float * data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_dense2_bias,{ 10 }, { 4 , 10 });
      std::copy(data, data + 40, tensor_dense2_biasbcast);
      delete [] data;
   }
}

std::vector<float> infer(float* tensor_input){

//--------- Gemm
   char op_0_transA = 'n';
   char op_0_transB = 'n';
   int op_0_m = 1;
   int op_0_n = 20;
   int op_0_k = 10;
   float op_0_alpha = 1;
   float op_0_beta = 1;
   int op_0_lda = 10;
   int op_0_ldb = 20;
   std::copy(tensor_dense_bias, tensor_dense_bias + 20, tensor_dense_output);
   BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_dense_weights, &op_0_ldb, tensor_input, &op_0_lda, &op_0_beta, tensor_dense_output, &op_0_n);
   ///--------Reshape operator

   std::copy( tensor_dense_output, tensor_dense_output + 20, tensor_reshape_output);

//--------- Gemm
   char op_2_transA = 'n';
   char op_2_transB = 'n';
   int op_2_m = 4;
   int op_2_n = 10;
   int op_2_k = 5;
   float op_2_alpha = 1;
   float op_2_beta = 1;
   int op_2_lda = 5;
   int op_2_ldb = 10;
   std::copy(tensor_dense1_biasbcast, tensor_dense1_biasbcast + 40, tensor_dense1_output);
   BLAS::sgemm_(&op_2_transB, &op_2_transA, &op_2_n, &op_2_m, &op_2_k, &op_2_alpha, tensor_dense1_weights, &op_2_ldb, tensor_reshape_output, &op_2_lda, &op_2_beta, tensor_dense1_output, &op_2_n);

//------ RELU
   for (int id = 0; id < 40 ; id++){
      tensor_relu_output[id] = ((tensor_dense1_output[id] > 0 )? tensor_dense1_output[id] : 0);
   }

//--------- Gemm
   char op_4_transA = 'n';
   char op_4_transB = 'n';
   int op_4_m = 4;
   int op_4_n = 10;
   int op_4_k = 10;
   float op_4_alpha = 1;
   float op_4_beta = 1;
   int op_4_lda = 10;
   int op_4_ldb = 10;
   std::copy(tensor_dense2_biasbcast, tensor_dense2_biasbcast + 40, tensor_dense2_output);
   BLAS::sgemm_(&op_4_transB, &op_4_transA, &op_4_n, &op_4_m, &op_4_k, &op_4_alpha, tensor_dense2_weights, &op_4_ldb, tensor_relu_output, &op_4_lda, &op_4_beta, tensor_dense2_output, &op_4_n);
   float op_5_alpha = 1;

//------ ELU 
   for (int id = 0; id < 40 ; id++){
      tensor_elu_output[id] = ((tensor_dense2_output[id] >= 0 )? tensor_dense2_output[id] : op_5_alpha * std::exp(tensor_dense2_output[id]) - 1);
   }
   return fTensor_elu_output;
}
};   // end of Session
} //TMVA_SOFIE_basic_model

#endif  // ROOT_TMVA_SOFIE_BASIC_MODEL
