
#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "library_float.h"

using namespace std ;
using namespace sci ;

extern float intToFloat(int32_t m);
extern void Softmax2(int32_t s1, int32_t s2, vector < vector < FPArray > >& inArr, vector < vector < FPArray > >& outArr);
extern void Ln(int32_t s1, vector < FPArray >& inArr, vector < FPArray >& outArr);
extern void Sigmoid(int32_t s1, vector<FPArray>& inArr, vector<FPArray>& outArr) ; 
extern void Tanh(int32_t s1, vector<FPArray>& inArr, vector<FPArray>& outArr) ; 
extern void getOutDer(int32_t s1, int32_t s2, vector < vector < FPArray > >& batchSoft, vector < vector < FPArray > >& lab, vector < vector < FPArray > >& der);
extern void MatMul(int32_t s1, int32_t s2, int32_t s3, vector < vector < FPArray > >& mat1, vector < vector < FPArray > >& mat2, vector < vector < FPArray > >& mat3);
extern void GemmAdd(int32_t s1, int32_t s2, vector < vector < FPArray > >& prod, vector < FPArray >& bias, vector < vector < FPArray > >& out);
extern void ElemWiseMul(int32_t s1, vector<FPArray>& arr1, vector<FPArray>& arr2, vector<FPArray>& outArr) ;
extern void ElemWiseAdd(int32_t s1, vector<FPArray>& arr1, vector<FPArray>& arr2, vector<FPArray>& outArr) ;
extern void ElemWiseDiv(int32_t s1, vector<FPArray>& arr1, vector<FPArray>& arr2, vector<FPArray>& outArr) ;
extern void SubtractOne(int32_t s1, vector<FPArray>& inArr, vector<FPArray>& outArr) ; 
extern void GemmAdd3(int32_t s1, int32_t s2, int32_t s3, vector<vector<vector<FPArray>>> &inArr, vector<FPArray> &bias, vector<vector<vector<FPArray>>> &outArr) ;
extern void dotProduct2(int32_t s1, int32_t s2, vector < vector < FPArray > >& arr1, vector < vector < FPArray > >& arr2, vector < FPArray >& outArr);
extern void Relu(int32_t s1, vector < FPArray >& inArr, vector < FPArray >& outArr, vector < BoolArray >& hotArr);
extern void getBiasDer(int32_t s1, int32_t s2, vector < vector < FPArray > >& der, vector < FPArray >& biasDer);
extern void IfElse(int32_t s1, vector < FPArray >& dat, vector < BoolArray >& hot, vector < FPArray >& out, bool flip);
extern void updateWeights(int32_t s, float lr, vector < FPArray >& bias, vector < FPArray >& der);
extern void updateWeightsAdam(int32_t s1, int32_t t, float lr, float beta1, float beta2, float eps, vector<FPArray>& inArr, vector<FPArray>& derArr, vector<FPArray>& mArr, vector<FPArray>& vArr) ;
extern void getLoss(int32_t m, vector < FPArray >& lossTerms, vector < FPArray >& loss);
extern void MatMul3(int32_t d1, int32_t d2, int32_t d3, int32_t d4, vector<vector<vector<FPArray>>> &arr1, vector<vector<vector<FPArray>>> &arr2, vector<vector<vector<FPArray>>> &arr3) ;

void preproc(uint32_t sz, auto v1, auto v2, auto v3, auto out) {
    auto add = make_vector_float(ALICE, sz) ;
    ElemWiseAdd(sz, v1, v2, add) ;
    ElemWiseDiv(sz, add, v3, out) ;
}

int main (int __argc, char **__argv) {
__init(__argc, __argv);

int32_t num_samples=30;
int32_t d2 = 270 ;
int32_t d3 = 6; 

uint32_t sz = num_samples*d2*d3 ;

auto vec1 = make_vector_float_rand(ALICE, sz) ;
auto vec2 = make_vector_float_rand(ALICE, sz) ;
auto den = make_vector_float_rand(ALICE, sz) ;
auto out = make_vector_float_rand(ALICE, sz) ;

auto start = clock_start();
float comm_start = __get_comm() ;

preproc(sz, vec1, vec2, den, out) ;

long long t = time_from(start);
float comm_end = __get_comm() ;
cout << "Total Time:\t" << t / (1000.0) << " ms" << endl;
cout << "Total comms:\t" << (comm_end - comm_start)/(1 << 20) ;

return 0;
}
