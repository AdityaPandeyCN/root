//Code generated automatically by TMVA for Inference of Model file [concat_model] at [Concat test mode] 

#ifndef ROOT_TMVA_SOFIE_CONCAT_MODEL
#define ROOT_TMVA_SOFIE_CONCAT_MODEL

#include <algorithm>
#include <vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_concat_model{
namespace BLAS{
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
struct Session {
// initialized tensors
std::vector<float> fTensor_dense3_weights = std::vector<float>(200);
float * tensor_dense3_weights = fTensor_dense3_weights.data();
std::vector<float> fTensor_dense3_bias = std::vector<float>(10);
float * tensor_dense3_bias = fTensor_dense3_bias.data();
std::vector<float> fTensor_dense2_bias = std::vector<float>(8);
float * tensor_dense2_bias = fTensor_dense2_bias.data();
std::vector<float> fTensor_dense2_weights = std::vector<float>(80);
float * tensor_dense2_weights = fTensor_dense2_weights.data();
std::vector<float> fTensor_dense1_bias = std::vector<float>(12);
float * tensor_dense1_bias = fTensor_dense1_bias.data();
std::vector<float> fTensor_dense1_weights = std::vector<float>(120);
float * tensor_dense1_weights = fTensor_dense1_weights.data();

//--- declare and allocate the intermediate tensors
std::vector<float> fTensor_concat_output = std::vector<float>(20);
float * tensor_concat_output = fTensor_concat_output.data();
std::vector<float> fTensor_relu2_output = std::vector<float>(8);
float * tensor_relu2_output = fTensor_relu2_output.data();
std::vector<float> fTensor_dense3_output = std::vector<float>(10);
float * tensor_dense3_output = fTensor_dense3_output.data();
std::vector<float> fTensor_dense2_output = std::vector<float>(8);
float * tensor_dense2_output = fTensor_dense2_output.data();
std::vector<float> fTensor_relu1_output = std::vector<float>(12);
float * tensor_relu1_output = fTensor_relu1_output.data();
std::vector<float> fTensor_dense1_output = std::vector<float>(12);
float * tensor_dense1_output = fTensor_dense1_output.data();


Session(std::string filename ="concat_model.dat") {

//--- reading weights from file
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("tmva-sofie failed to open file " + filename + " for input weights");
   }
   std::string tensor_name;
   size_t length;
   f >> tensor_name >> length;
   if (tensor_name != "tensor_dense3_weights" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_dense3_weights , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 200) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 200 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_dense3_weights[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_dense3_weights");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_dense3_bias" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_dense3_bias , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 10) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 10 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_dense3_bias[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_dense3_bias");
   }
   f >> tensor_name >> length;
   if (tensor_name != "tensor_dense2_bias" ) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor name; expected name is tensor_dense2_bias , read " + tensor_name;
      throw std::runtime_error(err_msg);
    }
   if (length != 8) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 8 , read " + std::to_string(length) ;
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
   if (length != 80) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 80 , read " + std::to_string(length) ;
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
   if (length != 12) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 12 , read " + std::to_string(length) ;
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
   if (length != 120) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is 120 , read " + std::to_string(length) ;
      throw std::runtime_error(err_msg);
    }
   for (size_t i = 0; i < length; ++i)
      f >> tensor_dense1_weights[i];
   if (f.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor tensor_dense1_weights");
   }
   f.close();

//---- allocate the intermediate dynamic tensors
}

std::vector<float> infer(float* tensor_input){

//--------- Gemm
   char op_0_transA = 'n';
   char op_0_transB = 'n';
   int op_0_m = 1;
   int op_0_n = 12;
   int op_0_k = 10;
   float op_0_alpha = 1;
   float op_0_beta = 1;
   int op_0_lda = 10;
   int op_0_ldb = 12;
   std::copy(tensor_dense1_bias, tensor_dense1_bias + 12, tensor_dense1_output);
   BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_dense1_weights, &op_0_ldb, tensor_input, &op_0_lda, &op_0_beta, tensor_dense1_output, &op_0_n);

//------ RELU
   for (int id = 0; id < 12 ; id++){
      tensor_relu1_output[id] = ((tensor_dense1_output[id] > 0 )? tensor_dense1_output[id] : 0);
   }

//--------- Gemm
   char op_2_transA = 'n';
   char op_2_transB = 'n';
   int op_2_m = 1;
   int op_2_n = 8;
   int op_2_k = 10;
   float op_2_alpha = 1;
   float op_2_beta = 1;
   int op_2_lda = 10;
   int op_2_ldb = 8;
   std::copy(tensor_dense2_bias, tensor_dense2_bias + 8, tensor_dense2_output);
   BLAS::sgemm_(&op_2_transB, &op_2_transA, &op_2_n, &op_2_m, &op_2_k, &op_2_alpha, tensor_dense2_weights, &op_2_ldb, tensor_input, &op_2_lda, &op_2_beta, tensor_dense2_output, &op_2_n);

//------ RELU
   for (int id = 0; id < 8 ; id++){
      tensor_relu2_output[id] = ((tensor_dense2_output[id] > 0 )? tensor_dense2_output[id] : 0);
   }

//--------- Concat
   std::copy(tensor_relu1_output, tensor_relu1_output+12, tensor_concat_output);
   std::copy(tensor_relu2_output, tensor_relu2_output+8, tensor_concat_output + 12);

//--------- Gemm
   char op_5_transA = 'n';
   char op_5_transB = 'n';
   int op_5_m = 1;
   int op_5_n = 10;
   int op_5_k = 20;
   float op_5_alpha = 1;
   float op_5_beta = 1;
   int op_5_lda = 20;
   int op_5_ldb = 10;
   std::copy(tensor_dense3_bias, tensor_dense3_bias + 10, tensor_dense3_output);
   BLAS::sgemm_(&op_5_transB, &op_5_transA, &op_5_n, &op_5_m, &op_5_k, &op_5_alpha, tensor_dense3_weights, &op_5_ldb, tensor_concat_output, &op_5_lda, &op_5_beta, tensor_dense3_output, &op_5_n);
   return fTensor_dense3_output;
}
};   // end of Session
} //TMVA_SOFIE_concat_model

#endif  // ROOT_TMVA_SOFIE_CONCAT_MODEL
