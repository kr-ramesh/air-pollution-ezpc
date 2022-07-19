#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "cleartext_library_float.h"

using namespace std ;

// template<typename T>
// vector<T> make_vector(size_t size) {
// return std::vector<T>(size) ;
// }

// template <typename T, typename... Args>
// auto make_vector(size_t first, Args... sizes)
// {
// auto inner = make_vector<T>(sizes...) ;
// return vector<decltype(inner)>(first, inner) ;
// }

extern float intToFloat(int32_t m);
extern void Softmax2(int32_t s1, int32_t s2, auto& inArr, auto& outArr);
extern void Ln(int32_t s1, auto& inArr, auto& outArr);
extern void getOutDer(int32_t s1, int32_t s2, auto& batchSoft, auto& lab, auto& der);
extern void MatMul(int32_t s1, int32_t s2, int32_t s3, auto& mat1, auto& mat2, auto& mat3);
extern void GemmAdd(int32_t s1, int32_t s2, auto& prod, auto& bias, auto& out);
extern void dotProduct2(int32_t s1, int32_t s2, auto& arr1, auto& arr2, auto& outArr);
extern void Relu(int32_t s1, auto& inArr, auto& outArr, auto& hotArr);
extern void getBiasDer(int32_t s1, int32_t s2, auto& der, auto& biasDer);
extern void IfElse(int32_t s1, auto& dat, auto& hot, auto& out, bool flip);
extern void updateWeights(int32_t s, float lr, auto& bias, auto& der);
extern void updateWeightsAdam(int32_t sz, int32_t t, float lr, float beta1, float beta2, float eps, auto inArr, auto derArr, auto& mArr, auto& vArr) ;
extern void getLoss(int32_t m, auto& lossTerms, auto& loss);
extern void computeMSELoss(int32_t m, int32_t s, auto& target, auto& fwdOut, auto& loss);

// void MatMul(int32_t s1, int32_t s2, int32_t s3, auto& mat1, auto& mat2, auto& mat3) {
// for (uint32_t i1 = 0; i1 < s1; i1++){
// for (uint32_t i3 = 0; i3 < s3; i3++){
// mat3[i1][i3] = 0. ;

// for (uint32_t i2 = 0; i2 < s2; i2++){
// mat3[i1][i3] = (mat3[i1][i3] + (mat1[i1][i2] * mat2[i2][i3])) ;

// }
// }
// }
// }

// void Relu(int32_t s1, auto& inArr, auto& outArr, auto& hotArr) {
// 	for (uint32_t i1 = 0; i1 < s1; i1++){
// 		hotArr[i1] = (inArr[i1] > 0.) ;
// 		outArr[i1] = hotArr[i1] ? inArr[i1] : 0. ;	
// 	}
// }

// void getBiasDer(int32_t s1, int32_t s2, auto& der, auto& biasDer) {
// 	for (uint32_t i2 = 0; i2 < s2; i2++){
// 		biasDer[i2] = der[0][i2] ;
// 		for (uint32_t i1 = 1; i1 < s1; i1++){
// 		biasDer[i2] = (biasDer[i2] + der[i1][i2]) ;

// 		}
// 	}
// }


// void GemmAdd(int32_t s1, int32_t s2, auto& prod, auto& bias, auto& out){
// for (uint32_t i1 = 0; i1 < s1; i1++){
// for (uint32_t i2 = 0; i2 < s2; i2++){
// float __tac_var17 = prod[i1][i2];

// float __tac_var18 = bias[i2] ;

// out[i1][i2]= (__tac_var17 + __tac_var18) ;

// }
// }
// }

void getSoftDer(int32_t s1, int32_t s2,auto& lab,  auto& batchSoft, auto& der){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
float __tac_var1 = batchSoft[i1][i2] ;

float __tac_var2 = lab[i1][i2] ;

der[i1][i2] = -(__tac_var2 - __tac_var1) ;

}
}
}

void Reassign2(int32_t s1, int32_t s2, auto& arr1, auto& arr2){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
arr2[i1][i2] = arr1[i1][i2] ;

}
}
}

void Transpose(int32_t s1, int32_t s2, int32_t s3, auto& inArr, auto& outArr){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
outArr[i1][i3][i2] = inArr[i1][i2][i3] ;

}
}
}
}

void Transpose2D(int32_t s1, int32_t s2, auto& inArr, auto& outArr){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
outArr[i2][i1] = inArr[i1][i2] ;

}
}
}

void Relu3(int32_t s1, int32_t s2, int32_t s3, auto& inArr, auto& outArr, auto& hotArr) {

int32_t sz = (s1 * s2 * s3) ;

auto hotFlat = make_vector<bool>(sz) ;
auto inFlat = make_vector<float>(sz) ;
auto outFlat = make_vector<float>(sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++) {
for (uint32_t i2 = 0; i2 < s2; i2++) {
for (uint32_t i3 = 0; i3 < s3; i3++) {

inFlat[i1*s2*s3 + i2*s3 + i3] = inArr[i1][i2][i3] ;

}
}
}

Relu(sz, inFlat, outFlat, hotFlat);

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){

outArr[i1][i2][i3] = outFlat[i1*s2*s3 + i2*s3 + i3] ;
hotArr[i1][i2][i3] = hotFlat[i1*s2*s3 + i2*s3 + i3] ;

}
}
}
}

void GemmAdd3(int32_t s1, int32_t s2, int32_t s3, auto& prod, auto& bias, auto& out){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
float __tac_var19 = prod[i1][i2][i3] ;

float __tac_var20 = bias[i2] ;

out[i1][i2][i3] = (__tac_var19 + __tac_var20) ;

}
}
}
}

void SubtractOne(int32_t d1, int32_t d2, auto& arr1, auto& arr2){
for (uint32_t j = 0; j < d1; j++){
for (uint32_t k = 0; k < d2; k++){
float __tac_var21 = 1. ;

float __tac_var22 = arr1[j][k] ;

arr2[j][k] = (__tac_var21 - __tac_var22) ;

}
}
}

void Ln2(int32_t s1, int32_t s2, auto& inArr, auto& outArr){
int32_t sz = (s1 * s2) ;

auto inFlat = make_vector<float>(sz) ;

auto outFlat = make_vector<float>(sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var23 = (i1 * s2) ;

int32_t __tac_var24 = (__tac_var23 + i2) ;

inFlat[__tac_var24] = inArr[i1][i2] ;

}
}
Ln(sz, inFlat, outFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var25 = (i1 * s2) ;

int32_t __tac_var26 = (__tac_var25 + i2) ;

outArr[i1][i2] = outFlat[__tac_var26] ;

}
}
}

void Relu2(int32_t s1, int32_t s2, auto& inArr, auto& outArr, auto& hotArr){
int32_t sz = (s1 * s2) ;

auto inArrFlat = make_vector<float>(sz) ;
auto outArrFlat = make_vector<float>(sz) ;
auto hotArrFlat = make_vector<bool>(sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var27 = (i1 * s2) ;

int32_t __tac_var28 = (__tac_var27 + i2) ;

inArrFlat[__tac_var28] = inArr[i1][i2] ;

}
}
Relu(sz, inArrFlat, outArrFlat, hotArrFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var29 = (i1 * s2) ;

int32_t __tac_var30 = (__tac_var29 + i2) ;

outArr[i1][i2] = outArrFlat[__tac_var30] ;

int32_t __tac_var31 = __tac_var29 ;

int32_t __tac_var32 = __tac_var30 ;

hotArr[i1][i2] = hotArrFlat[__tac_var30] ;

}
}
}

// void updateWeights(int32_t s, float lr, auto& bias, auto& der) {
// 	for (uint32_t i1 = 0; i1 < s; i1++){
// 		bias[i1] = bias[i1] - (lr*der[i1]) ;
// 	}
// }

void updateWeights2(int32_t s1, int32_t s2, float lr, auto& weight, auto& der){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
    weight[i1][i2] = weight[i1][i2] - (lr*der[i1][i2]) ; 
    }
    }
}

void updateWeightsAdam2(int32_t s1, int32_t s2, int32_t t, float lr, float beta1, float beta2, float eps, vector<vector<float>>& weight, vector<vector<float>>& der, vector<vector<float>>& mArr, vector<vector<float>>& vArr) {
int32_t sz = (s1 * s2) ;

auto weightFlat = make_vector<float>(sz) ;
auto derFlat = make_vector<float>(sz) ;
auto vArrFlat = make_vector<float>(sz) ;
auto mArrFlat = make_vector<float>(sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
weightFlat[((i1 * s2) + i2)] = weight[i1][i2] ;
derFlat[((i1 * s2) + i2)] = der[i1][i2] ;
vArrFlat[((i1 * s2) + i2)]= vArr[i1][i2];
mArrFlat[((i1 * s2) + i2)]= mArr[i1][i2];
}
}

updateWeightsAdam(sz, t, lr, beta1, beta2, eps, weightFlat, derFlat, mArrFlat, vArrFlat);

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
weight[i1][i2] = weightFlat[((i1 * s2) + i2)] ;
vArr[i1][i2] = vArrFlat[i1*s2 + i2] ;
mArr[i1][i2] = mArrFlat[i1*s2 + i2] ;
}
}
}

void IfElse2(int32_t s1, int32_t s2, auto& dat, auto& hot, auto& out, bool flip){
int32_t sz = (s1 * s2) ;

auto datFlat = make_vector<float>(sz) ;
auto hotFlat = make_vector<bool>(sz) ;
auto outFlat = make_vector<float>(sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var39 = (i1 * s2) ;

int32_t __tac_var40 = (__tac_var39 + i2) ;

datFlat[__tac_var40] = dat[i1][i2] ;

int32_t __tac_var41 = __tac_var39 ;

int32_t __tac_var42 = __tac_var40 ;

hotFlat[__tac_var42] = hot[i1][i2] ;

}
}
IfElse(sz, datFlat, hotFlat, outFlat, flip);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
int32_t __tac_var43 = (i1 * s2) ;

int32_t __tac_var44 = (__tac_var43 + i2) ;

out[i1][i2] = outFlat[__tac_var44] ;

}
}
}

void Sigmoid2(int32_t s1, int32_t s2, auto& inArr, auto& outArr){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
outArr[i1][i2]=1/(1+exp(-inArr[i1][i2]));
}
}
}

void Tanh2(int32_t s1, int32_t s2, auto& inArr, auto& outArr){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
float t1= exp(inArr[i1][i2]);
float t2= exp(-inArr[i1][i2]);
outArr[i1][i2]=(t1-t2)/(t2+t1);
}
}
}

//ElemWiseMul
void ElemProd(int32_t s1, int32_t s2, auto& arr1, auto& arr2, auto& outArr){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
float __tac_var45 = arr1[i1][i2] ;
float __tac_var46 = arr2[i1][i2] ;
outArr[i1][i2] = (__tac_var45 * __tac_var46) ;

}
}
}

void ElemProd3(int32_t s1, int32_t s2, int32_t s3, auto& arr1, auto& arr2, auto& outArr){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
float __tac_var47 = arr1[i1][i2][i3] ;
float __tac_var48 = arr2[i1][i2][i3] ;
outArr[i1][i2][i3] = (__tac_var47 * __tac_var48) ;

}
}
}
}

void computeCELoss(int32_t m, int32_t s2, auto& labels, auto& batchSoft, auto& loss){
auto batchLn = make_vector<float>(m, s2) ;
auto lossTerms = make_vector<float>(m) ;

Ln2(m, s2, batchSoft, batchLn);
dotProduct2(m, s2, batchLn, labels, lossTerms);
getLoss(m, lossTerms, loss);
}

void Assign3(int32_t s1, int32_t s2, int32_t s3, auto& arr2, auto& arr1){
for (uint32_t l = 0; l < s1; l++){
for (uint32_t j = 0; j < s2; j++){
arr1[s3][l][j] = arr2[l][j] ;

}
}
}

void AssignCopy2D(int32_t s1, int32_t s2, auto& arr1, auto& arr2){
for (uint32_t l = 0; l < s1; l++){
for (uint32_t j = 0; j < s2; j++){
arr2[l][j] = arr1[l][j] ;
}
}
}

void AssignInputs1D(int32_t s2, int32_t s3, auto& arr1, auto& arr2){
for (uint32_t j = 0; j < s2; j++){
arr2[j][s3] = arr1[0][j] ;

}
}

void ReverseAssign3(int32_t s1, int32_t s2, int32_t s3, auto& arr1, auto& arr2){
for (uint32_t l = 0; l < s1; l++){
for (uint32_t j = 0; j < s2; j++){
arr2[l][j] = arr1[s3][l][j] ;
}
}
}

void MatMul3(int32_t d1, int32_t d2, int32_t d3, int32_t d4, auto& arr1, auto& arr2, auto& arr3){
auto a1 = make_vector<float>(d2, d3) ;
auto a2 = make_vector<float>(d3, d4) ;
auto a3 = make_vector<float>(d2, d4) ;

for (uint32_t i0 = 0; i0 < d1; i0++) {
ReverseAssign3(d2, d3, i0, arr1, a1) ;
ReverseAssign3(d3, d4, i0, arr2, a2) ;
MatMul(d2, d3, d4, a1, a2, a3) ;
Assign3(d2, d4, i0, a3, arr3) ;
}
}

//ElemWiseAdd
void Additive2D(int32_t d1, int32_t d2, auto& arr1, auto& arr2, auto& arr3){
for (uint32_t j = 0; j < d1; j++){
for (uint32_t k = 0; k < d2; k++){
float __tac_var49 = arr2[j][k] ;
float __tac_var50 = arr1[j][k] ;
arr3[j][k] = (__tac_var49 + __tac_var50) ;
}
}
}

//GemmAdd
void Additive2DBias(int32_t d1, int32_t d2, auto& arr1, auto& arr2){
for (uint32_t j = 0; j < d1; j++){
for (uint32_t k = 0; k < d2; k++){
float __tac_var51 = arr2[k] ;
float __tac_var52 = arr1[j][k] ;
arr2[k] = (__tac_var51 + __tac_var52) ;
}
}
}

// void AssignSampleFromData3D(int32_t s1, int32_t s2, int32_t index, auto& arr1, auto& arr2){
// for (uint32_t l = 0; l < s1; l++){
// for (uint32_t j = 0; j < s2; j++){
// arr2[0][l][j] = arr1[index][l][j] ;
// }
// }
// }

// void AssignSampleFromData2D(int32_t s1, int32_t index, auto& arr1, auto& arr2){
// for (uint32_t l = 0; l < s1; l++){
// arr2[0][l] = arr1[index][l] ;
// }
// }

void AssignGates(int32_t d1, int32_t d2, int32_t d3, auto& arr1, auto& arr2, auto& arr3, auto& arr4, auto& arr5){
for (uint32_t j = 0; j < d1; j++) {
for (uint32_t k = 0; k < d2; k++) {

arr5[j][k] = arr1[j][k] ;
int32_t __tac_var55 = (d2 + k) ;
arr5[j][__tac_var55] = arr2[j][k] ;
int32_t __tac_var56 = (2 * d2) ;
int32_t __tac_var57 = (__tac_var56 + k) ;
arr5[j][__tac_var57] = arr3[j][k] ;
int32_t __tac_var58 = (3 * d2) ;
int32_t __tac_var59 = (__tac_var58 + k) ;
arr5[j][__tac_var59] = arr4[j][k] ;

}
}
}

void FixedAdjacencyGraph(int32_t d1, int32_t d2, int32_t d3, int32_t d4, auto& features, auto& lastnodes, auto& kernelarr, auto& bias, auto& finalarr, auto& neighbours, auto& reluoutputs){
auto featurest = make_vector<float>(d1, d3, d2) ;
auto neighbourst = make_vector<float>(d1, d2, d3) ;
auto outputarr = make_vector<float>(d1, d2, d4) ;
auto ht = make_vector<bool>(d1, d2, d4) ;

Transpose(d1, d2, d3, features, featurest);
MatMul3(d1, d3, d2, d2, featurest, lastnodes, neighbours);
Transpose(d1, d3, d2, neighbours, neighbourst);
MatMul3(d1, d2, d3, d4, neighbourst, kernelarr, outputarr);
GemmAdd3(d1, d2, d4, outputarr, bias, outputarr);
Relu3(d1, d2, d4, outputarr, finalarr, ht);
reluoutputs = ht[0] ;

}

void LSTMStep(int32_t t, int32_t inputdim, int32_t hiddendim, int32_t dim1, int32_t dim3, auto& Inputs, auto& HInp, auto& CellStates, auto& Filter, auto& RecurrentFilter, auto& Bias, auto& FullHt, auto& FullIt, auto& FullFt, auto& FullGt, auto& FullOt, auto& FullCt, auto& FullXt){

auto Z1 = make_vector<float>(dim1, dim3) ;
auto Z2 = make_vector<float>(dim1, dim3) ;
auto Z = make_vector<float>(dim1, dim3) ;
auto O = make_vector<float>(dim1, hiddendim) ;
auto C = make_vector<float>(dim1, hiddendim) ;
auto F = make_vector<float>(dim1, hiddendim) ;
auto I = make_vector<float>(dim1, hiddendim) ;
auto Term1 = make_vector<float>(dim1, hiddendim) ;
auto Term2 = make_vector<float>(dim1, hiddendim) ;
auto CellStates2 = make_vector<float>(dim1, hiddendim) ;

Assign3(dim1, hiddendim, t, HInp, FullHt);
Assign3(dim1, inputdim, t, Inputs, FullXt);
MatMul(dim1, inputdim, dim3, Inputs, Filter, Z1);
MatMul(dim1, hiddendim, dim3, HInp, RecurrentFilter, Z2);

for (uint32_t i = 0; i < dim1; i++) {
for (uint32_t j = 0; j < dim3; j++) {
    float __tac_var60 = Z1[i][j] ;
    float __tac_var61 = Z2[i][j] ;
    Z[i][j] = (__tac_var60 + __tac_var61) ;
}
}
GemmAdd(dim1, dim3, Z, Bias, Z);
for (uint32_t i = 0; i < dim1; i++){
for (uint32_t j = 0; j < hiddendim; j++){
I[i][j] = Z[i][j] ;

int32_t __tac_var62 = (hiddendim + j) ;

F[i][j] = Z[i][__tac_var62] ;

int32_t __tac_var63 = (2 * hiddendim) ;

int32_t __tac_var64 = (__tac_var63 + j) ;

C[i][j] = Z[i][__tac_var64] ;

int32_t __tac_var65 = (3 * hiddendim) ;

int32_t __tac_var66 = (__tac_var65 + j) ;

O[i][j] = Z[i][__tac_var66] ;

}
}

Sigmoid2(dim1, hiddendim, I, I);
Sigmoid2(dim1, hiddendim, F, F);
Tanh2(dim1, hiddendim, C, C);
ElemProd(dim1, hiddendim, CellStates, F, Term1);
ElemProd(dim1, hiddendim, I, C, Term2);
for (uint32_t i = 0; i < dim1; i++){
for (uint32_t j = 0; j < hiddendim; j++){
float __tac_var67 = Term1[i][j] ;

float __tac_var68 = Term2[i][j] ;

CellStates[i][j] = (__tac_var67 + __tac_var68) ;

}
}
Sigmoid2(dim1, hiddendim, O, O);
Tanh2(dim1, hiddendim, CellStates, CellStates2);
ElemProd(dim1, hiddendim, O, CellStates2, HInp);

Assign3(dim1, hiddendim, t, O, FullOt);
Assign3(dim1, hiddendim, t, F, FullFt);
Assign3(dim1, hiddendim, t, I, FullIt);
Assign3(dim1, hiddendim, t, C, FullGt);
int32_t __tac_var69 = (t + 1) ;

Assign3(dim1, hiddendim, __tac_var69, CellStates, FullCt);
}

void LSTM3D(int32_t numunits, int32_t idim, int32_t hdim, int32_t d1, int32_t d3, auto& AllInputs, auto& HState, auto& CellState, auto& Fil, auto& RecFil, auto& biasunit, auto& FullHt, auto& FullIt, auto& FullFt, auto& FullGt, auto& FullOt, auto& FullCt, auto& FullXt){
auto InputExample = make_vector<float>(d1, idim) ;
for (uint32_t iter = 0; iter < numunits; iter++) {
    for (uint32_t s0 = 0; s0 < d1; s0++){
    for (uint32_t l = 0; l < idim; l++) {
        InputExample[s0][l] = AllInputs[s0][iter][l] ;
    }
    }
    LSTMStep(iter, idim, hdim, d1, d3, InputExample, HState, CellState, Fil, RecFil, biasunit, FullHt, FullIt, FullFt, FullGt, FullOt, FullCt, FullXt);
    }
}


void forward(int32_t d1, int32_t d2, int32_t d3, int32_t d4, auto& features, auto& lastnodes, auto& kernelarr, auto& gcnbias, auto& lastnodes2, auto& kernelarr2, auto& gcnbias2, int32_t unitstotal, int32_t idim, int32_t hdim, int32_t dim1, int32_t dim3, auto& hstates, auto& cellstates, auto& lstmkernel, auto& reclstmkernel, auto& lstmbias, auto& denselayer, auto& bias4, auto& FullHt, auto& FullIt, auto& FullFt, auto& FullGt, auto& FullOt, auto& FullCt, auto& FullXt, auto& neight2, auto& feat2, auto& neigh1, auto& feat1, auto& reluoutputs1, auto& reluoutputs2, auto& finaloutput){

auto totalhidden = make_vector<float>(unitstotal, dim1, hdim) ;
auto finalarr = make_vector<float>(d1, d2, d4) ;
auto finalarr2 = make_vector<float>(d1, d2, d4) ;
auto neighbours = make_vector<float>(d1, d3, d2) ;
auto neighbours2 = make_vector<float>(d1, d4, d2) ;
auto finalarr3 = make_vector<float>(d1, d4, d2) ;
auto neighbourst2 = make_vector<float>(d1, d2, d4) ;

FixedAdjacencyGraph(d1, d2, d3, d4, features, lastnodes, kernelarr, gcnbias, finalarr, neighbours, reluoutputs1);
ReverseAssign3(d2, d3, 0, features, feat1);
ReverseAssign3(d3, d2, 0, neighbours, neigh1);

FixedAdjacencyGraph(d1, d2, d4, d4, finalarr, lastnodes2, kernelarr2, gcnbias2, finalarr2, neighbours2, reluoutputs2);
Transpose(d1, d4, d2, neighbours2, neighbourst2);
ReverseAssign3(d2, d4, 0, neighbourst2, neight2);
ReverseAssign3(d2, d4, 0, finalarr, feat2);
Transpose(d1, d2, d4, finalarr2, finalarr3);

Assign3(d1, hdim, 0, cellstates, FullCt);
LSTM3D(unitstotal, idim, hdim, dim1, dim3, finalarr3, hstates, cellstates, lstmkernel, reclstmkernel, lstmbias, FullHt, FullIt, FullFt, FullGt, FullOt, FullCt, FullXt);
Assign3(d1, hdim, d4, hstates, FullHt);

MatMul(dim1, hdim, d2, hstates, denselayer, finaloutput);
GemmAdd(dim1, d2, finaloutput, bias4, finaloutput);
Sigmoid2(dim1, d2, finaloutput, finaloutput);

}

void backward(
	int32_t d1, int32_t d2, int32_t gcn1dim3, int32_t gcn2dim3, int32_t hdim, int32_t hdim4, int32_t totaltimesteps, 
	auto& layer1W, auto& layer1b, auto& hiddenstates, auto& TotalIt, auto& TotalFt, auto& TotalGt, auto& TotalOt, auto& TotalCt, auto& TotalXt, 
	auto& batchSoft, auto& lab, auto& Fil, auto& RecFil, auto& LSTMBias, auto& neighbourst2, auto& kernel2, auto& features2, auto& A2, auto& neighbours1, auto& kernel1, auto& features1, auto& A1, auto& bias1, auto& bias2, auto& reluoutputs1,auto& reluoutputs2,
	auto& m1,auto& v1,auto& m2,auto& v2,auto& m3,auto& v3,auto& m4,auto& v4,auto& m5,auto& v5,auto& m6,auto& v6,auto& m7,auto& v7,auto& m8,auto& v8,auto& m9,auto& v9,auto& m10,auto& v10,auto& m11,auto& v11, int32_t total_t, float learnrate) {

auto layer1Der = make_vector<float>(d1, d2) ;
auto dh = make_vector<float>(d1, hdim) ;
auto layer1WTranspose = make_vector<float>(d2, hdim) ;
auto layer1InReshaped = make_vector<float>(hdim, d1) ;
auto layer1WDerReshaped = make_vector<float>(hdim, d2) ;
auto layer1WDer = make_vector<float>(d2, hdim) ;
auto layerFinalHidden = make_vector<float>(d1, hdim) ;
auto hiddenT = make_vector<float>(d1, hdim) ;
auto layer1bDer = make_vector<float>(d2) ;
auto DX = make_vector<float>(d2, totaltimesteps) ;
auto DFilSum = make_vector<float>(d2, hdim4) ;
auto DRecFilSum = make_vector<float>(hdim, hdim4) ;
auto DBiasSum = make_vector<float>(hdim4) ;
auto dbias1 = make_vector<float>(d2) ;
auto dbias2 = make_vector<float>(d2) ;

ReverseAssign3(d1, hdim, totaltimesteps, hiddenstates, layerFinalHidden);
getSoftDer(d1, d2, lab, batchSoft, layer1Der);
ElemProd(d1, d2, layer1Der, batchSoft, layer1Der);
SubtractOne(d1, d2, batchSoft, batchSoft);
ElemProd(d1, d2, layer1Der, batchSoft, layer1Der);
Transpose2D(d1, hdim, layerFinalHidden, layer1InReshaped);
MatMul(hdim, d1, d2, layer1InReshaped, layer1Der, layer1WDerReshaped);

Transpose2D(hdim, d2, layer1WDerReshaped, layer1WDer);
Transpose2D(hdim, d2, layer1W, layer1WTranspose);
MatMul(d1, d2, hdim, layer1Der, layer1WTranspose, dh);
getBiasDer(d1, d2, layer1Der, layer1bDer);

auto dNextC = make_vector<float>(d1, hdim) ;

for (uint32_t t = 0; t < totaltimesteps; t++){
    auto temp = make_vector<float>(d1, hdim) ;
    auto Ot = make_vector<float>(d1, hdim) ;
    auto It = make_vector<float>(d1, hdim) ;
    auto Ft = make_vector<float>(d1, hdim) ;
    auto Gt = make_vector<float>(d1, hdim) ;
    auto Ct = make_vector<float>(d1, hdim) ;
    auto DOt = make_vector<float>(d1, hdim) ;
    auto DIt = make_vector<float>(d1, hdim) ;
    auto DFt = make_vector<float>(d1, hdim) ;
    auto DGt = make_vector<float>(d1, hdim) ;
    auto DCt = make_vector<float>(d1, hdim) ;
    auto Xt = make_vector<float>(d1, d2) ;
    auto PrevC = make_vector<float>(d1, hdim) ;
    auto NextF = make_vector<float>(d1, hdim) ;
    auto DGates = make_vector<float>(d1, hdim4) ;
    auto FilTranspose = make_vector<float>(hdim4, d2) ;
    auto RecFilTranspose = make_vector<float>(hdim4, hdim) ;
    auto XtTranspose = make_vector<float>(d2, d1) ;
    auto hiddenTTranspose = make_vector<float>(hdim, d1) ;
    auto dXt = make_vector<float>(d1, d2) ;
    auto DFil = make_vector<float>(d2, hdim4) ;
    auto DRecFil = make_vector<float>(hdim, hdim4) ;

int32_t __tac_var70 = (totaltimesteps - t) ;
int32_t __tac_var71 = (__tac_var70 - 1) ;

ReverseAssign3(d1, hdim, __tac_var71, hiddenstates, hiddenT);

int32_t __tac_var72 = __tac_var70 ;
int32_t __tac_var73 = __tac_var71 ;

ReverseAssign3(d1, hdim, __tac_var71, TotalOt, Ot);

int32_t __tac_var74 = __tac_var70 ;
int32_t __tac_var75 = __tac_var71 ;

ReverseAssign3(d1, hdim, __tac_var71, TotalIt, It);

int32_t __tac_var76 = __tac_var70 ;
int32_t __tac_var77 = __tac_var71 ;

ReverseAssign3(d1, hdim, __tac_var71, TotalFt, Ft);

int32_t __tac_var78 = __tac_var70 ;
int32_t __tac_var79 = __tac_var71 ;

ReverseAssign3(d1, hdim, __tac_var71, TotalGt, Gt);

int32_t __tac_var80 = __tac_var70 ;
int32_t __tac_var81 = __tac_var71 ;

ReverseAssign3(d1, hdim, __tac_var70, TotalCt, Ct);

int32_t __tac_var82 = __tac_var70 ;

ReverseAssign3(d1, hdim, __tac_var71, TotalCt, PrevC);
int32_t __tac_var84 = __tac_var70 ;

if(t!=0) {
    ReverseAssign3(d1, hdim, __tac_var70, TotalFt, NextF);
}

int32_t __tac_var85 = __tac_var70 ;

ReverseAssign3(d1, d2, __tac_var71, TotalXt, Xt);
Reassign2(d1, hdim, Ot, DOt);
ElemProd(d1, hdim, dh, DOt, DOt);
Tanh2(d1, hdim, Ct, Ct);
ElemProd(d1, hdim, Ct, Ct, temp);
SubtractOne(d1, hdim, temp, temp);
ElemProd(d1, hdim, Ot, temp, DCt);
ElemProd(d1, hdim, DCt, dh, DCt);
ElemProd(d1, hdim, dNextC, NextF, temp);
Additive2D(d1, hdim, temp, DCt, DCt);

for (uint32_t i1 = 0; i1 < d1; i1++)
    {
    for (uint32_t i2 = 0; i2 < hdim; i2++){
        dNextC[i1][i2] = DCt[i1][i2] ;
        }
        }

ElemProd(d1, hdim, DCt, It, DGt);
ElemProd(d1, hdim, DGt, Gt, DIt);
ElemProd(d1, hdim, Gt, Gt, temp);
SubtractOne(d1, hdim, temp, temp);
ElemProd(d1, hdim, DGt, temp, DGt);


SubtractOne(d1, hdim, It, temp);
ElemProd(d1, hdim, DIt, temp, DIt);
ElemProd(d1, hdim, DCt, PrevC, DFt);
ElemProd(d1, hdim, DFt, Ft, DFt);
SubtractOne(d1, hdim, Ft, temp);
ElemProd(d1, hdim, DFt, temp, DFt);
SubtractOne(d1, hdim, Ot, temp);
ElemProd(d1, hdim, DOt, temp, DOt);
ElemProd(d1, hdim, DOt, Ct, DOt);


AssignGates(d1, hdim, hdim4, DIt, DFt, DGt, DOt, DGates);

Transpose2D(d2, hdim4, Fil, FilTranspose);
Transpose2D(hdim, hdim4, RecFil, RecFilTranspose);
MatMul(d1, hdim4, d2, DGates, FilTranspose, dXt);
MatMul(d1, hdim4, hdim, DGates, RecFilTranspose, dh);
Transpose2D(d1, d2, Xt, XtTranspose);
Transpose2D(d1, hdim, hiddenT, hiddenTTranspose);

MatMul(d2, d1, hdim4, XtTranspose, DGates, DFil);
MatMul(hdim, d1, hdim4, hiddenTTranspose, DGates, DRecFil);
Additive2D(d2, hdim4, DFil, DFilSum, DFilSum);
Additive2D(hdim, hdim4, DRecFil, DRecFilSum, DRecFilSum);
Additive2DBias(d1, hdim4, DGates, DBiasSum);
int32_t __tac_var87 = __tac_var70 ;

int32_t __tac_var88 = __tac_var71 ;

AssignInputs1D(d2, __tac_var71, dXt, DX);
}

auto dkernelarr2 = make_vector<float>(totaltimesteps, gcn2dim3) ;

auto dkernelarr = make_vector<float>(gcn1dim3, gcn2dim3) ;

auto kernelarr2 = make_vector<float>(totaltimesteps, gcn2dim3) ;

auto kernelarr = make_vector<float>(gcn1dim3, gcn2dim3) ;

ReverseAssign3(totaltimesteps, gcn2dim3, 0, kernel2, kernelarr2);
ReverseAssign3(gcn1dim3, gcn2dim3, 0, kernel1, kernelarr);

auto lastnodes2 = make_vector<float>(d2, d2) ;

auto lastnodes = make_vector<float>(d2, d2) ;
ReverseAssign3(d2, d2, 0, A1, lastnodes);
ReverseAssign3(d2, d2, 0, A2, lastnodes2);

ReverseAssign3(totaltimesteps, gcn2dim3, 0, kernel2, kernelarr2);
ReverseAssign3(gcn1dim3, gcn2dim3, 0, kernel1, kernelarr);

auto DXTranspose = make_vector<float>(totaltimesteps, d2) ;

auto dneighbours2 = make_vector<float>(gcn2dim3, d2) ;

auto dneighbours1 = make_vector<float>(gcn1dim3, d2) ;

auto dLastnodes2 = make_vector<float>(d2, d2) ;

auto dLastnodes1 = make_vector<float>(d2, d2) ;

auto dfeatures2t = make_vector<float>(gcn2dim3, d2) ;

auto dfeatures2 = make_vector<float>(d2, gcn2dim3) ;

auto reluoutputs1t = make_vector<float>(gcn2dim3, d2) ;

//Reverse DX in the final version <----- VERY IMPORTANT otherwise it will be upside down
ElemProd(d2, totaltimesteps, DX, reluoutputs2, DX);
Transpose2D(d2, totaltimesteps, DX, DXTranspose);

MatMul(totaltimesteps, d2, gcn2dim3, DXTranspose, neighbourst2, dkernelarr2);
getBiasDer(d2, totaltimesteps, DX, dbias1);

MatMul(gcn2dim3, totaltimesteps, d2, kernelarr2, DXTranspose, dneighbours2);
MatMul(d2, gcn2dim3, d2, features2, dneighbours2, dLastnodes2);
MatMul(gcn2dim3, d2, d2, dneighbours2, lastnodes2, dfeatures2t);

Transpose2D(d2, gcn2dim3, reluoutputs1, reluoutputs1t);
ElemProd(gcn2dim3, d2, dfeatures2t, reluoutputs1t, dfeatures2t);
getBiasDer(gcn2dim3, d2, dfeatures2t, dbias2);
Transpose2D(gcn2dim3, d2, dfeatures2t, dfeatures2);
MatMul(gcn1dim3, d2, gcn2dim3, neighbours1, dfeatures2, dkernelarr);
MatMul(gcn1dim3, gcn2dim3, d2, kernelarr, dfeatures2t, dneighbours1);
MatMul(d2, gcn1dim3, d2, features1, dneighbours1, dLastnodes1);

updateWeightsAdam2(gcn1dim3, gcn2dim3, total_t, learnrate, 0.999, 0.999, 1e-7, kernelarr, dkernelarr, m1, v1);
updateWeightsAdam2(totaltimesteps, gcn2dim3, total_t, learnrate, 0.999, 0.999, 1e-7, kernelarr2, dkernelarr2, m2, v2);
updateWeightsAdam2(d2, d2, total_t, learnrate, 0.999, 0.999, 1e-7, dLastnodes1, lastnodes, m3, v3);
updateWeightsAdam2(d2, d2, total_t, learnrate, 0.999, 0.999, 1e-7, dLastnodes2, lastnodes2, m4, v4);
updateWeightsAdam(d2, total_t, learnrate, 0.999, 0.999, 1e-7, dbias1, bias1, m5, v5);
updateWeightsAdam(d2, total_t, learnrate, 0.999, 0.999, 1e-7, dbias2, bias2, v6, v6);
updateWeightsAdam2(d2, hdim4, total_t, learnrate, 0.999, 0.999, 1e-7, Fil, DFilSum, m7, v7);
updateWeightsAdam2(hdim, hdim4, total_t, learnrate, 0.999, 0.999, 1e-7, RecFil, DRecFilSum, m8, v8);
updateWeightsAdam(hdim4, total_t, learnrate, 0.999, 0.999, 1e-7, LSTMBias, DBiasSum, m9, v9);
updateWeightsAdam2(hdim, total_t, d2, learnrate, 0.999, 0.99, 1e-7, layer1W, layer1WDerReshaped, m10, v10);
updateWeightsAdam(d2, total_t, learnrate, 0.999, 0.999, 1e-7, layer1b, layer1bDer, m11, v11);

}

void AssignSampleFromData3D(int32_t s0, int32_t s1, int32_t s2, int32_t index, auto& arr1, auto& arr2){
for (uint32_t k = 0; k < s0; k++){    
for (uint32_t l = 0; l < s1; l++){
for (uint32_t j = 0; j < s2; j++){
arr2[k][l][j] = arr1[index][l][j] ;
}
}
index++;
}
}

void AssignSampleFromData2D(int32_t s0, int32_t s1, int32_t index, auto& arr1, auto& arr2){
for (uint32_t k = 0; k < s0; k++){       
for (uint32_t l = 0; l < s1; l++){
arr2[k][l] = arr1[index][l] ;
}
index++;
}
}


int main (int __argc, char **__argv) {

float lr= 0.001;
int32_t num_samples=30;
int32_t d1 = 19 ;
int32_t d2 = 270 ;
int32_t d3 = 6; 
int32_t d4 = 4 ;
int32_t hdim = 4 ;
int32_t gatesdim = d4*4;

auto inp = make_vector<float>(num_samples, d2, d3) ;
cout << ("Input inp:") << endl ;
float *__tmp_in_inp = new float[1] ;
for (uint32_t i0 = 0; i0 < num_samples; i0++){
for (uint32_t i1 = 0; i1 < d2; i1++){
for (uint32_t i2 = 0; i2 < d3; i2++){
cin >> __tmp_in_inp[0];
inp[i0][i1][i2] = __tmp_in_inp[0] ;

}
}
}
delete[] __tmp_in_inp ;


auto A1 = make_vector<float>(d1, d2, d2) ;
cout << ("Input A1:") << endl ;
float *__tmp_in_A1 = new float[1] ;

for (uint32_t i0 = 0; i0 < d1; i0++){
for (uint32_t i1 = 0; i1 < d2; i1++){
for (uint32_t i2 = 0; i2 < d2; i2++){
cin >> __tmp_in_A1[0];
A1[i0][i1][i2] = __tmp_in_A1[0] ;

}
}
}
delete[] __tmp_in_A1 ;

auto kernel1 = make_vector<float>(d1, d3, d4) ;

cout << ("Input kernel1:") << endl ;

float *__tmp_in_kernel1 = new float[1] ;

for (uint32_t i0 = 0; i0 < d1; i0++){
for (uint32_t i1 = 0; i1 < d3; i1++){
for (uint32_t i2 = 0; i2 < d4; i2++){
cin >> __tmp_in_kernel1[0];
kernel1[i0][i1][i2] = __tmp_in_kernel1[0] ;

}
}
}
delete[] __tmp_in_kernel1 ;

auto bias1 = make_vector<float>(d2) ;

cout << ("Input bias1:") << endl ;

float *__tmp_in_bias1 = new float[1] ;

for (uint32_t i0 = 0; i0 < d2; i0++){
cin >> __tmp_in_bias1[0];
bias1[i0] = __tmp_in_bias1[0] ;

}
delete[] __tmp_in_bias1 ;

auto A2 = make_vector<float>(d1, d2, d2) ;

cout << ("Input A2:") << endl ;

float *__tmp_in_A2 = new float[1] ;

for (uint32_t i0 = 0; i0 < d1; i0++){
for (uint32_t i1 = 0; i1 < d2; i1++){
for (uint32_t i2 = 0; i2 < d2; i2++){
cin >> __tmp_in_A2[0];
A2[i0][i1][i2] = __tmp_in_A2[0] ;

}
}
}
delete[] __tmp_in_A2 ;

auto kernel2 = make_vector<float>(d1, d4, d4) ;

cout << ("Input kernel2:") << endl ;

float *__tmp_in_kernel2 = new float[1] ;

for (uint32_t i0 = 0; i0 < d1; i0++){
for (uint32_t i1 = 0; i1 < d4; i1++){
for (uint32_t i2 = 0; i2 < d4; i2++){
cin >> __tmp_in_kernel2[0];
kernel2[i0][i1][i2] = __tmp_in_kernel2[0] ;

}
}
}
delete[] __tmp_in_kernel2 ;

auto bias2 = make_vector<float>(d2) ;

cout << ("Input bias2:") << endl ;

float *__tmp_in_bias2 = new float[1] ;

for (uint32_t i0 = 0; i0 < d2; i0++){
cin >> __tmp_in_bias2[0];
bias2[i0] = __tmp_in_bias2[0] ;

}
delete[] __tmp_in_bias2 ;

auto hidden = make_vector<float>(d1, d4) ;

cout << ("Input hidden:") << endl ;

float *__tmp_in_hidden = new float[1] ;

for (uint32_t i0 = 0; i0 < d1; i0++){
for (uint32_t i1 = 0; i1 < d4; i1++){
cin >> __tmp_in_hidden[0];
hidden[i0][i1] = __tmp_in_hidden[0] ;

}
}
delete[] __tmp_in_hidden ;

auto cell = make_vector<float>(d1, d4) ;

cout << ("Input cell:") << endl ;

float *__tmp_in_cell = new float[1] ;

for (uint32_t i0 = 0; i0 < d1; i0++){
for (uint32_t i1 = 0; i1 < d4; i1++){
cin >> __tmp_in_cell[0];
cell[i0][i1] = __tmp_in_cell[0] ;

}
}
delete[] __tmp_in_cell ;

auto k = make_vector<float>(d2, gatesdim) ;

cout << ("Input k:") << endl ;

float *__tmp_in_k = new float[1] ;

for (uint32_t i0 = 0; i0 < d2; i0++){
for (uint32_t i1 = 0; i1 < gatesdim; i1++){
cin >> __tmp_in_k[0];
k[i0][i1] = __tmp_in_k[0] ;

}
}

auto reck = make_vector<float>(d4, gatesdim) ;

cout << ("Input reck:") << endl ;

float *__tmp_in_reck = new float[1] ;

for (uint32_t i0 = 0; i0 < d4; i0++){
for (uint32_t i1 = 0; i1 < gatesdim; i1++){
cin >> __tmp_in_reck[0];
reck[i0][i1] = __tmp_in_reck[0] ;

}
}

auto lstmbias = make_vector<float>(gatesdim) ;

cout << ("Input lstmbias:") << endl ;

float *__tmp_in_lstmbias = new float[1] ;

for (uint32_t i0 = 0; i0 < gatesdim; i0++){
cin >> __tmp_in_lstmbias[0];
lstmbias[i0] = __tmp_in_lstmbias[0] ;

}

delete[] __tmp_in_k ;

delete[] __tmp_in_reck ;

delete[] __tmp_in_lstmbias ;

auto dense = make_vector<float>(d4, d2) ;

cout << ("Input dense:") << endl ;

float *__tmp_in_dense = new float[1] ;

for (uint32_t i0 = 0; i0 < d4; i0++){
for (uint32_t i1 = 0; i1 < d2; i1++){
cin >> __tmp_in_dense[0];
dense[i0][i1] = __tmp_in_dense[0] ;

}
}
delete[] __tmp_in_dense ;

auto bias4 = make_vector<float>(d2) ;

cout << ("Input bias4:") << endl ;

float *__tmp_in_bias4 = new float[1] ;

for (uint32_t i0 = 0; i0 < d2; i0++){
cin >> __tmp_in_bias4[0];
bias4[i0] = __tmp_in_bias4[0] ;

}
auto labs = make_vector<float>(num_samples, d2) ;

cout << ("Input labs:") << endl ;

for (uint32_t i1 = 0; i1 < num_samples; i1++){
for (uint32_t i0 = 0; i0 < d2; i0++){
cin >> __tmp_in_bias4[0];
labs[i1][i0] = __tmp_in_bias4[0] ;
}
}
delete[] __tmp_in_bias4 ;


auto neight2 = make_vector<float>(d2, d4) ;
auto feat2 = make_vector<float>(d2, d4) ;
auto neigh1 = make_vector<float>(d3, d2) ;
auto feat1 = make_vector<float>(d2, d3) ;
auto reluoutputs1 = make_vector<bool>(d2, d4) ;
auto reluoutputs2 = make_vector<bool>(d2, d4) ;

auto m1 = make_vector<float>(d3, d4) ;
auto v1 = make_vector<float>(d3, d4) ;
auto m2 = make_vector<float>(d4, d4) ;
auto v2 = make_vector<float>(d4, d4) ;
auto m3 = make_vector<float>(d2, d2) ;
auto v3 = make_vector<float>(d2, d2) ;
auto m4 = make_vector<float>(d2, d2) ;
auto v4 = make_vector<float>(d2, d2) ;
auto m5 = make_vector<float>(d2) ;
auto v5 = make_vector<float>(d2) ;
auto m6 = make_vector<float>(d2) ;
auto v6 = make_vector<float>(d2) ;
auto m7 = make_vector<float>(d2, gatesdim) ;
auto v7 = make_vector<float>(d2, gatesdim) ;
auto m8 = make_vector<float>(hdim, gatesdim) ;
auto v8 = make_vector<float>(hdim, gatesdim) ;
auto m9 = make_vector<float>(gatesdim) ;
auto v9 = make_vector<float>(gatesdim) ;
auto m10= make_vector<float>(hdim, d2) ;
auto v10 = make_vector<float>(hdim, d2) ;
auto m11 = make_vector<float>(d2) ;
auto v11 = make_vector<float>(d2) ;

int total_timesteps=0;

cout<<"Starting neural network!"<<endl;

for(uint32_t iterations=0; iterations<20; iterations++)
{
    int num_assign=0;
    for (uint32_t ind = 0; ind < num_samples; ind=ind+d1){
        if(num_samples-ind<d1)
        num_assign=num_samples-ind;
        else
        num_assign=d1;

        auto FullHt = make_vector<float>((d4 + 1), num_assign, d4) ;
        auto FullIt = make_vector<float>(d4, num_assign, d4) ;
        auto FullFt = make_vector<float>(d4, num_assign, d4) ;
        auto FullGt = make_vector<float>(d4, num_assign, d4) ;
        auto FullOt = make_vector<float>(d4, num_assign, d4) ;
        auto FullCt = make_vector<float>((d4 + 1), num_assign, d4) ;
        auto FullXt = make_vector<float>(d4, num_assign, d2)  ;
        auto inputpoint = make_vector<float>(num_assign, d2, d3) ;
        auto labels = make_vector<float>(num_assign, d2) ;
        auto finaloutput = make_vector<float>(num_assign, d2) ;
        

        cout<<"Assignment"<<endl;
        AssignSampleFromData3D(num_assign, d2, d3, ind, inp, inputpoint);
        cout<<"Second Assignment"<<endl;
        AssignSampleFromData2D(num_assign, d2, ind, labs, labels);

        cout<<"FORWARD"<<endl;
        forward(num_assign, d2, d3, d4, inputpoint, A1, kernel1, bias1, A2, kernel2, bias2, d4, d2, hdim, num_assign, gatesdim, hidden, cell, k, reck, lstmbias, dense, bias4, FullHt, FullIt, FullFt, FullGt, FullOt, FullCt, FullXt, neight2, feat2, neigh1, feat1, reluoutputs1, reluoutputs2, finaloutput);
        //computeLoss(d1, d2, labels, finaloutput);
        cout<<"BACKWARD"<<endl;
        total_timesteps+=1;
        backward(num_assign, d2, d3, 4, 4, gatesdim, 4, dense, bias4, FullHt, FullIt, FullFt, FullGt, FullOt, FullCt, FullXt, finaloutput, labels, k, reck, lstmbias, neight2, kernel2, feat2, A2, neigh1, kernel1, feat1, A2, bias1, bias2, reluoutputs1, reluoutputs2, m1, v1, m2, v2, m3, v3, m4, v4, m5, v5, m6, v6, m7, v7, m8, v8, m9, v9, m10, v10, m11, v11, total_timesteps, lr);
    }
}
}