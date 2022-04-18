
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
extern void SubtractOne(int32_t s1, vector<FPArray>& inArr, vector<FPArray>& outArr) ; 

extern void GemmAdd3(int32_t s1, int32_t s2, int32_t s3,
    vector<vector<vector<FPArray>>> &inArr, 
    vector<FPArray> &bias, 
    vector<vector<vector<FPArray>>> &outArr) ;

extern void dotProduct2(int32_t s1, int32_t s2, vector < vector < FPArray > >& arr1, vector < vector < FPArray > >& arr2, vector < FPArray >& outArr);
extern void Relu(int32_t s1, vector < FPArray >& inArr, vector < FPArray >& outArr, vector < BoolArray >& hotArr);
extern void getBiasDer(int32_t s1, int32_t s2, vector < vector < FPArray > >& der, vector < FPArray >& biasDer);
extern void IfElse(int32_t s1, vector < FPArray >& dat, vector < BoolArray >& hot, vector < FPArray >& out, bool flip);
extern void updateWeights(int32_t s, float lr, vector < FPArray >& bias, vector < FPArray >& der);
extern void getLoss(int32_t m, vector < FPArray >& lossTerms, vector < FPArray >& loss);

void ElemWiseMul2(int32_t s1, int32_t s2, vector<vector<FPArray>>& arr1, vector<vector<FPArray>>& arr2, vector<vector<FPArray>>& outArr) {
    int sz = s1*s2 ;

    vector<FPArray> arr1_flat = make_vector_float(ALICE, sz) ;
    vector<FPArray> arr2_flat = make_vector_float(ALICE, sz) ;
    vector<FPArray> outarr_flat = make_vector_float(ALICE, sz) ;

    for (int i = 0 ; i < s1 ; i++) {
        for (int j = 0 ; j < s2 ; j++) {
            arr1_flat[i*s2 + j] = arr1[i][j] ;
            arr2_flat[i*s2 + j] = arr2[i][j] ;
        }
    }

    ElemWiseMul(sz, arr1_flat, arr2_flat, outarr_flat) ;

    for (int i = 0 ; i < s1 ; i++) {
        for (int j = 0 ; j < s2 ; j++) {
            outArr[i][j] = outarr_flat[i*s2 + j] ;
        }
    }
}

void ElemWiseAdd2(int32_t s1, int32_t s2, vector<vector<FPArray>>& arr1, vector<vector<FPArray>>& arr2, vector<vector<FPArray>>& outArr) {
    int sz = s1*s2 ;

    vector<FPArray> arr1_flat = make_vector_float(ALICE, sz) ;
    vector<FPArray> arr2_flat = make_vector_float(ALICE, sz) ;
    vector<FPArray> outarr_flat = make_vector_float(ALICE, sz) ;

    for (int i = 0 ; i < s1 ; i++) {
        for (int j = 0 ; j < s2 ; j++) {
            arr1_flat[i*s2 + j] = arr1[i][j] ;
            arr2_flat[i*s2 + j] = arr2[i][j] ;
        }
    }

    ElemWiseAdd(sz, arr1_flat, arr2_flat, outarr_flat) ;

    for (int i = 0 ; i < s1 ; i++) {
        for (int j = 0 ; j < s2 ; j++) {
            outArr[i][j] = outarr_flat[i*s2 + j] ;
        }
    }
}

void AssignInputs1D(int32_t s2, int32_t s3, auto& arr1, auto& arr2){
for (uint32_t j = 0; j < s2; j++){
arr2[j][s3] = arr1[0][j] ;
}
}

void AssignGates(int32_t d1, int32_t d2, int32_t d3, auto& arr1, auto& arr2, auto& arr3, auto& arr4, auto& arr5){
for (uint32_t j = 0; j < d1; j++){
for (uint32_t k = 0; k < d2; k++){
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



const int32_t totalsteps = 16;
const int32_t dim2 = 1989;
const int32_t g2d3 = 16;
const int32_t g1d3 = 12;
int32_t count_var=0;

void Sigmoid2(int32_t s1, int32_t s2, vector < vector < FPArray > >& inArr, vector < vector < FPArray > >& outArr){
int32_t sz = (s1 * s2) ;

vector < FPArray > inFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > outFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
inFlat[((i1 * s2) + i2)] = inArr[i1][i2] ;

}
}
Sigmoid(sz, inFlat, outFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
outArr[i1][i2] = outFlat[((i1 * s2) + i2)] ;

}
}
}

void SubtractOne2(int32_t s1, int32_t s2, vector<vector<FPArray>>& inArr, vector <vector<FPArray>>& outArr){
int32_t sz = (s1 * s2) ;

vector < FPArray > inFlat = make_vector_float(ALICE, sz) ;
vector < FPArray > outFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
inFlat[((i1 * s2) + i2)] = inArr[i1][i2] ;

}
}
SubtractOne(sz, inFlat, outFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
outArr[i1][i2] = outFlat[((i1 * s2) + i2)] ;

}
}
}

void updateWeights2(int32_t s1, int32_t s2, float lr, vector < vector < FPArray > >& weight, vector < vector < FPArray > >& der){
int32_t sz = (s1 * s2) ;

vector < FPArray > weightFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > derFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
weightFlat[((i1 * s2) + i2)] = weight[i1][i2] ;

derFlat[((i1 * s2) + i2)] = der[i1][i2] ;

}
}
updateWeights(sz, lr, weightFlat, derFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
weight[i1][i2] = weightFlat[((i1 * s2) + i2)] ;

}
}
}

void Tanh2(int32_t s1, int32_t s2, vector < vector < FPArray > >& inArr, vector < vector < FPArray > >& outArr){
int32_t sz = (s1 * s2) ;

vector < FPArray > inFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > outFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
inFlat[((i1 * s2) + i2)] = inArr[i1][i2] ;

}
}
Tanh(sz, inFlat, outFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
outArr[i1][i2] = outFlat[((i1 * s2) + i2)] ;

}
}
}

void ReverseAssign3(int32_t s1, int32_t s2, int32_t s3, auto& arr1, auto& arr2){
for (uint32_t l = 0; l < s1; l++){
for (uint32_t j = 0; j < s2; j++){
arr2[l][j] = arr1[s3][l][j] ;

}
}
}

void Assign3(int32_t s1, int32_t s2, int32_t s3, auto& arr2, auto& arr1){
for (uint32_t l = 0; l < s1; l++){
for (uint32_t j = 0; j < s2; j++){
arr1[s3][l][j] = arr2[l][j] ;

}
}
}


void IfElse2(int32_t s1, int32_t s2, vector < vector < FPArray > >& dat, vector < vector < BoolArray > >& hot, vector < vector < FPArray > >& out, bool flip){
int32_t sz = (s1 * s2) ;

vector < FPArray > datFlat = make_vector_float(ALICE, sz) ;

vector < BoolArray > hotFlat = make_vector_bool(ALICE, sz) ;

vector < FPArray > outFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
datFlat[((i1 * s2) + i2)] = dat[i1][i2] ;

hotFlat[((i1 * s2) + i2)] = hot[i1][i2] ;

}
}
IfElse(sz, datFlat, hotFlat, outFlat, flip);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
out[i1][i2] = outFlat[((i1 * s2) + i2)] ;

}
}
}


void MatMul3(int32_t d1, int32_t d2, int32_t d3, int32_t d4, auto& arr1, auto& arr2, auto& arr3){
auto a1 = make_vector_float(ALICE, d2, d3) ;
auto a2 = make_vector_float(ALICE, d3, d4) ;
auto a3 = make_vector_float(ALICE, d2, d4) ;

for (uint32_t i0 = 0; i0 < d1; i0++){
    ReverseAssign3(d2, d3, i0, arr1, a1);
    ReverseAssign3(d3, d4, i0, arr2, a2);
    MatMul(d2, d3, d4, a1, a2, a3);
    Assign3(d2, d4, i0, a3, arr3);
}

}

void Transpose3D(int32_t s1, int32_t s2, int32_t s3, auto& inArr, auto& outArr){
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

void Reassign2(int32_t s1, int32_t s2, auto& arr1, auto& arr2){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
arr2[i1][i2] = arr1[i1][i2] ;

}
}
}

void Relu3(int32_t s1, int32_t s2, int32_t s3, auto& inArr, auto& outArr, auto& hotArr){

int32_t sz = (s1 * s2 * s3) ;

auto hotFlat = make_vector_bool(ALICE, sz) ;
auto inFlat = make_vector_float(ALICE, sz) ;
auto outFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
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


void AssignSampleFromData3D(int32_t s1, int32_t s2, int32_t index, auto& arr1, auto& arr2){
for (uint32_t l = 0; l < s1; l++){
for (uint32_t j = 0; j < s2; j++){
arr2[0][l][j] = arr1[index][l][j] ;
}
}
}

void AssignSampleFromData2D(int32_t s1, int32_t index, auto& arr1, auto& arr2){
for (uint32_t l = 0; l < s1; l++){
arr2[0][l] = arr1[index][l] ;
}
}

void FixedAdjacencyGraph(int32_t d1, int32_t d2, int32_t d3, int32_t d4, auto& features, auto& lastnodes, auto& kernelarr, auto& bias, auto& finalarr, auto& neighbours, auto& reluoutputs){
auto featurest = make_vector_float(ALICE, d1, d3, d2) ;
auto neighbourst = make_vector_float(ALICE, d1, d2, d3) ;
auto outputarr = make_vector_float(ALICE, d1, d2, d4) ;
auto ht = make_vector_bool(ALICE, d1, d2, d4) ;

Transpose3D(d1, d2, d3, features, featurest);
MatMul3(d1, d3, d2, d2, featurest, lastnodes, neighbours);
Transpose3D(d1, d3, d2, neighbours, neighbourst);
MatMul3(d1, d2, d3, d4, neighbourst, kernelarr, outputarr);
GemmAdd3(d1, d2, d4, outputarr, bias, outputarr);

Relu3(d1, d2, d4, outputarr, finalarr, ht);
reluoutputs = ht[0] ;
}

void LSTMStep(int32_t t, int32_t inputdim, int32_t hiddendim, int32_t dim1, int32_t dim3, auto& Inputs, auto& HInp, auto& CellStates, auto& Filter, auto& RecurrentFilter, auto& Bias, auto& FullHt, auto& FullIt, auto& FullFt, auto& FullGt, auto& FullOt, auto& FullCt, auto& FullXt){

auto Z1 = make_vector_float(ALICE, dim1, dim3) ;
auto Z2 = make_vector_float(ALICE, dim1, dim3) ;
auto Z = make_vector_float(ALICE, dim1, dim3) ;
auto O = make_vector_float(ALICE, dim1, hiddendim) ;
auto C = make_vector_float(ALICE, dim1, hiddendim) ;
auto F = make_vector_float(ALICE, dim1, hiddendim) ;
auto Idem = make_vector_float(ALICE, dim1, hiddendim) ;
auto Term1 = make_vector_float(ALICE, dim1, hiddendim) ;
auto Term2 = make_vector_float(ALICE, dim1, hiddendim) ;
auto CellStates2 = make_vector_float(ALICE, dim1, hiddendim) ;

Assign3(dim1, hiddendim, t, HInp, FullHt);
Assign3(dim1, inputdim, t, Inputs, FullXt);
MatMul(dim1, inputdim, dim3, Inputs, Filter, Z1);
MatMul(dim1, hiddendim, dim3, HInp, RecurrentFilter, Z2);

// for (uint32_t i = 0; i < dim1; i++) {
// for (uint32_t j = 0; j < dim3; j++) {
//     float __tac_var60 = Z1[i][j] ;
//     float __tac_var61 = Z2[i][j] ;
//     Z[i][j] = (__tac_var60 + __tac_var61) ;
// }
// }
ElemWiseAdd2(dim1, dim3, Z1, Z2, Z) ;

GemmAdd(dim1, dim3, Z, Bias, Z);

for (uint32_t i = 0; i < dim1; i++){
for (uint32_t j = 0; j < hiddendim; j++){
Idem[i][j] = Z[i][j] ;

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

Sigmoid2(dim1, hiddendim, Idem, Idem);
Sigmoid2(dim1, hiddendim, F, F);
Tanh2(dim1, hiddendim, C, C);
ElemWiseMul2(dim1, hiddendim, CellStates, F, Term1);
ElemWiseMul2(dim1, hiddendim, Idem, C, Term2);

// for (uint32_t i = 0; i < dim1; i++){
// for (uint32_t j = 0; j < hiddendim; j++){
// float __tac_var67 = Term1[i][j] ;

// float __tac_var68 = Term2[i][j] ;

// CellStates[i][j] = (__tac_var67 + __tac_var68) ;

// }
// }
ElemWiseAdd2(dim1, hiddendim, Term1, Term2, CellStates) ;

Sigmoid2(dim1, hiddendim, O, O);
Tanh2(dim1, hiddendim, CellStates, CellStates2);
ElemWiseMul2(dim1, hiddendim, O, CellStates2, HInp);

Assign3(dim1, hiddendim, t, O, FullOt);
Assign3(dim1, hiddendim, t, F, FullFt);
Assign3(dim1, hiddendim, t, Idem, FullIt);
Assign3(dim1, hiddendim, t, C, FullGt);
Assign3(dim1, hiddendim, t+1, CellStates, FullCt);
}

void LSTM3D(int32_t numunits, int32_t idim, int32_t hdim, int32_t d1, int32_t d3, auto& AllInputs, auto& HState, auto& CellState, auto& Fil, auto& RecFil, auto& biasunit, auto& FullHt, auto& FullIt, auto& FullFt, auto& FullGt, auto& FullOt, auto& FullCt, auto& FullXt){
auto InputExample = make_vector_float(ALICE, 1, idim) ;
for (uint32_t iter = 0; iter < numunits; iter++) {
    for (uint32_t l = 0; l < idim; l++) {
        InputExample[0][l] = AllInputs[0][iter][l] ;
    }
    LSTMStep(iter, idim, hdim, d1, d3, InputExample, HState, CellState, Fil, RecFil, biasunit, FullHt, FullIt, FullFt, FullGt, FullOt, FullCt, FullXt);
    }
}


void forward(int32_t d1, int32_t d2, int32_t d3, int32_t d4, auto& features, auto& lastnodes, auto& kernelarr, auto& gcnbias, auto& lastnodes2, auto& kernelarr2, auto& gcnbias2, int32_t unitstotal, int32_t idim, int32_t hdim, int32_t dim1, int32_t dim3, auto& hstates, auto& cellstates, auto& lstmkernel, auto& reclstmkernel, auto& lstmbias, auto& denselayer, auto& bias4, auto& FullHt, auto& FullIt, auto& FullFt, auto& FullGt, auto& FullOt, auto& FullCt, auto& FullXt, auto& neight2, auto& feat2, auto& neigh1, auto& feat1, auto& reluoutputs1, auto& reluoutputs2, auto& finaloutput){

auto totalhidden = make_vector_float(ALICE, unitstotal, dim1, hdim) ;
auto finalarr = make_vector_float(ALICE, d1, d2, d4) ;
auto finalarr2 = make_vector_float(ALICE, d1, d2, d4) ;
auto neighbours = make_vector_float(ALICE, d1, d3, d2) ;
auto neighbours2 = make_vector_float(ALICE, d1, d4, d2) ;
auto finalarr3 = make_vector_float(ALICE, d1, d4, d2) ;
auto neighbourst2 = make_vector_float(ALICE, d1, d2, d4) ;

FixedAdjacencyGraph(d1, d2, d3, d4, features, lastnodes, kernelarr, gcnbias, finalarr, neighbours, reluoutputs1);
ReverseAssign3(d2, d3, 0, features, feat1);
ReverseAssign3(d3, d2, 0, neighbours, neigh1);

FixedAdjacencyGraph(d1, d2, d4, d4, finalarr, lastnodes2, kernelarr2, gcnbias2, finalarr2, neighbours2, reluoutputs2);
Transpose3D(d1, d4, d2, neighbours2, neighbourst2);
ReverseAssign3(d2, d4, 0, neighbourst2, neight2);
ReverseAssign3(d2, d4, 0, finalarr, feat2);
Transpose3D(d1, d2, d4, finalarr2, finalarr3);

Assign3(d1, hdim, 0, cellstates, FullCt);
LSTM3D(unitstotal, idim, hdim, dim1, dim3, finalarr3, hstates, cellstates, lstmkernel, reclstmkernel, lstmbias, FullHt, FullIt, FullFt, FullGt, FullOt, FullCt, FullXt);
Assign3(d1, hdim, d4, hstates, FullHt);

MatMul(dim1, hdim, d2, hstates, denselayer, finaloutput);
GemmAdd(dim1, d2, finaloutput, bias4, finaloutput);
Sigmoid2(dim1, d2, finaloutput, finaloutput);

}

void backward(int32_t d1, int32_t d2, int32_t gcn1dim3, int32_t gcn2dim3, int32_t hdim, int32_t hdim4, int32_t totaltimesteps, auto& layer1W, auto& layer1b, auto& hiddenstates, auto& TotalIt, auto& TotalFt, auto& TotalGt, auto& TotalOt, auto& TotalCt, auto& TotalXt, auto& batchSoft, auto& lab, auto& Fil, auto& RecFil, auto& LSTMBias, auto& neighbourst2, auto& kernel2, auto& features2, auto& A2, auto& neighbours1, auto& kernel1, auto& features1, auto& A1, auto& bias1, auto& bias2, auto& reluoutputs1,auto& reluoutputs2){

    auto layer1Der = make_vector_float(ALICE, d1, d2) ;
    auto dh = make_vector_float(ALICE, d1, hdim) ;
    auto layer1WTranspose = make_vector_float(ALICE, d2, hdim) ;
    auto layer1InReshaped = make_vector_float(ALICE, hdim, d1) ;
    auto layer1WDerReshaped = make_vector_float(ALICE, hdim, d2) ;
    auto layer1WDer = make_vector_float(ALICE, d2, hdim) ;
    auto layerFinalHidden = make_vector_float(ALICE, d1, hdim) ;
    auto hiddenT = make_vector_float(ALICE, d1, hdim) ;
    auto layer1bDer = make_vector_float(ALICE, d2) ;
    auto DX = make_vector_float(ALICE, d2, totaltimesteps) ;
    auto DFilSum = make_vector_float(ALICE, d2, hdim4) ;
    auto DRecFilSum = make_vector_float(ALICE, hdim, hdim4) ;
    auto DBiasSum = make_vector_float(ALICE, hdim4) ;
    auto dbias1 = make_vector_float(ALICE, d2) ;
    auto dbias2 = make_vector_float(ALICE, d2) ;

ReverseAssign3(d1, hdim, totaltimesteps, hiddenstates, layerFinalHidden);
getOutDer(d1, d2, batchSoft, lab, layer1Der);
ElemWiseMul2(d1, d2, layer1Der, batchSoft, layer1Der);
SubtractOne2(d1, d2, batchSoft, batchSoft);
ElemWiseMul2(d1, d2, layer1Der, batchSoft, layer1Der);
Transpose2D(d1, hdim, layerFinalHidden, layer1InReshaped);
MatMul(hdim, d1, d2, layer1InReshaped, layer1Der, layer1WDerReshaped);

Transpose2D(hdim, d2, layer1WDerReshaped, layer1WDer);
Transpose2D(hdim, d2, layer1W, layer1WTranspose);
MatMul(d1, d2, hdim, layer1Der, layer1WTranspose, dh);
getBiasDer(d1, d2, layer1Der, layer1bDer);

auto dNextC = make_vector_float(ALICE, d1, hdim) ;

for (uint32_t t = 0; t < totaltimesteps; t++) {
    auto temp = make_vector_float(ALICE, d1, hdim) ;
    auto Ot = make_vector_float(ALICE, d1, hdim) ;
    auto It = make_vector_float(ALICE, d1, hdim) ;
    auto Ft = make_vector_float(ALICE, d1, hdim) ;
    auto Gt = make_vector_float(ALICE, d1, hdim) ;
    auto Ct = make_vector_float(ALICE, d1, hdim) ;
    auto DOt = make_vector_float(ALICE, d1, hdim) ;
    auto DIt = make_vector_float(ALICE, d1, hdim) ;
    auto DFt = make_vector_float(ALICE, d1, hdim) ;
    auto DGt = make_vector_float(ALICE, d1, hdim) ;
    auto DCt = make_vector_float(ALICE, d1, hdim) ;
    auto Xt = make_vector_float(ALICE, d1, d2) ;
    auto PrevC = make_vector_float(ALICE, d1, hdim) ;
    auto NextF = make_vector_float(ALICE, d1, hdim) ;
    auto DGates = make_vector_float(ALICE, d1, hdim4) ;
    auto FilTranspose = make_vector_float(ALICE, hdim4, d2) ;
    auto RecFilTranspose = make_vector_float(ALICE, hdim4, hdim) ;
    auto XtTranspose = make_vector_float(ALICE, d2, d1) ;
    auto hiddenTTranspose = make_vector_float(ALICE, hdim, d1) ;
    auto dXt = make_vector_float(ALICE, d1, d2) ;
    auto DFil = make_vector_float(ALICE, d2, hdim4) ;
    auto DRecFil = make_vector_float(ALICE, hdim, hdim4) ;

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
ElemWiseMul2(d1, hdim, dh, DOt, DOt);
Tanh2(d1, hdim, Ct, Ct);
ElemWiseMul2(d1, hdim, Ct, Ct, temp);
SubtractOne2(d1, hdim, temp, temp);
ElemWiseMul2(d1, hdim, Ot, temp, DCt);
ElemWiseMul2(d1, hdim, DCt, dh, DCt);
ElemWiseMul2(d1, hdim, dNextC, NextF, temp);
ElemWiseAdd2(d1, hdim, temp, DCt, DCt);

for (uint32_t i1 = 0; i1 < d1; i1++)
    {
    for (uint32_t i2 = 0; i2 < hdim; i2++){
        dNextC[i1][i2] = DCt[i1][i2] ;
        }
        }

ElemWiseMul2(d1, hdim, DCt, It, DGt);
ElemWiseMul2(d1, hdim, DGt, Gt, DIt);
ElemWiseMul2(d1, hdim, Gt, Gt, temp);
SubtractOne2(d1, hdim, temp, temp);
ElemWiseMul2(d1, hdim, DGt, temp, DGt);


SubtractOne2(d1, hdim, It, temp);
ElemWiseMul2(d1, hdim, DIt, temp, DIt);
ElemWiseMul2(d1, hdim, DCt, PrevC, DFt);
ElemWiseMul2(d1, hdim, DFt, Ft, DFt);
SubtractOne2(d1, hdim, Ft, temp);
ElemWiseMul2(d1, hdim, DFt, temp, DFt);
SubtractOne2(d1, hdim, Ot, temp);
ElemWiseMul2(d1, hdim, DOt, temp, DOt);
ElemWiseMul2(d1, hdim, DOt, Ct, DOt);

AssignGates(d1, hdim, hdim4, DIt, DFt, DGt, DOt, DGates);

Transpose2D(d2, hdim4, Fil, FilTranspose);
Transpose2D(hdim, hdim4, RecFil, RecFilTranspose);
MatMul(d1, hdim4, d2, DGates, FilTranspose, dXt);
MatMul(d1, hdim4, hdim, DGates, RecFilTranspose, dh);
Transpose2D(d1, d2, Xt, XtTranspose);
Transpose2D(d1, hdim, hiddenT, hiddenTTranspose);

MatMul(d2, d1, hdim4, XtTranspose, DGates, DFil);
MatMul(hdim, d1, hdim4, hiddenTTranspose, DGates, DRecFil);
ElemWiseAdd2(d2, hdim4, DFil, DFilSum, DFilSum);
ElemWiseAdd2(hdim, hdim4, DRecFil, DRecFilSum, DRecFilSum);
getBiasDer(d1, hdim4, DGates, DBiasSum);

int32_t __tac_var87 = __tac_var70 ;
int32_t __tac_var88 = __tac_var71 ;

AssignInputs1D(d2, __tac_var71, dXt, DX);

}

auto dkernelarr2 = make_vector_float(ALICE, totaltimesteps, gcn2dim3) ;
auto dkernelarr = make_vector_float(ALICE, gcn1dim3, gcn2dim3) ;
auto kernelarr2 = make_vector_float(ALICE, totaltimesteps, gcn2dim3) ;
auto kernelarr = make_vector_float(ALICE, gcn1dim3, gcn2dim3) ;

ReverseAssign3(totaltimesteps, gcn2dim3, 0, kernel2, kernelarr2);
ReverseAssign3(gcn1dim3, gcn2dim3, 0, kernel1, kernelarr);

auto lastnodes2 = make_vector_float(ALICE, d2, d2) ;

auto lastnodes = make_vector_float(ALICE, d2, d2) ;

ReverseAssign3(d2, d2, 0, A1, lastnodes);
ReverseAssign3(d2, d2, 0, A2, lastnodes2);
ReverseAssign3(totaltimesteps, gcn2dim3, 0, kernel2, kernelarr2);
ReverseAssign3(gcn1dim3, gcn2dim3, 0, kernel1, kernelarr);

auto DXTranspose = make_vector_float(ALICE, totaltimesteps, d2) ;
auto dneighbours2 = make_vector_float(ALICE, gcn2dim3, d2) ;
auto dneighbours1 = make_vector_float(ALICE, gcn1dim3, d2) ;
auto dLastnodes2 = make_vector_float(ALICE, d2, d2) ;
auto dLastnodes1 = make_vector_float(ALICE, d2, d2) ;
auto dfeatures2t = make_vector_float(ALICE, gcn2dim3, d2) ;
auto dfeatures2 = make_vector_float(ALICE, d2, gcn2dim3) ;
auto reluoutputs1t = make_vector_bool(ALICE, gcn2dim3, d2) ;

//Reverse DX in the final version <----- VERY IMPORTANT otherwise it will be upside down
IfElse2(d2, totaltimesteps, DX, reluoutputs2, DX, true);
Transpose2D(d2, totaltimesteps, DX, DXTranspose);

MatMul(totaltimesteps, d2, gcn2dim3, DXTranspose, neighbourst2, dkernelarr2);
getBiasDer(d2, totaltimesteps, DX, dbias1);

MatMul(gcn2dim3, totaltimesteps, d2, kernelarr2, DXTranspose, dneighbours2);
MatMul(d2, gcn2dim3, d2, features2, dneighbours2, dLastnodes2);
MatMul(gcn2dim3, d2, d2, dneighbours2, lastnodes2, dfeatures2t);

Transpose2D(d2, gcn2dim3, reluoutputs1, reluoutputs1t);
IfElse2(gcn2dim3, d2, dfeatures2t, reluoutputs1t, dfeatures2t, true);
getBiasDer(gcn2dim3, d2, dfeatures2t, dbias2);
Transpose2D(gcn2dim3, d2, dfeatures2t, dfeatures2);
MatMul(gcn1dim3, d2, gcn2dim3, neighbours1, dfeatures2, dkernelarr);
MatMul(gcn1dim3, gcn2dim3, d2, kernelarr, dfeatures2t, dneighbours1);
MatMul(d2, gcn1dim3, d2, features1, dneighbours1, dLastnodes1);

updateWeights2(gcn1dim3, gcn2dim3, 0.01, kernelarr, dkernelarr);
updateWeights2(totaltimesteps, gcn2dim3, 0.01, kernelarr2, dkernelarr2);
updateWeights2(d2, d2, 0.01, dLastnodes1, lastnodes);
updateWeights2(d2, d2, 0.01, dLastnodes2, lastnodes2);
updateWeights(d2, 0.01, dbias1, bias1);
updateWeights(d2, 0.01, dbias2, bias2);
updateWeights2(d2, hdim4, 0.01, Fil, DFilSum);
updateWeights2(hdim, hdim4, 0.01, RecFil, DRecFilSum);
updateWeights(hdim4, 0.01, LSTMBias, DBiasSum);
updateWeights2(hdim, d2, 0.01, layer1W, layer1WDerReshaped);
updateWeights(d2, 0.01, layer1b, layer1bDer);

}


int main (int __argc, char **__argv) {

int32_t num_samples=26;
int32_t d1 = 1 ;
int32_t d2 = 270 ;
int32_t d3 = 3; 
int32_t d4 = 16 ;
int32_t hdim = 16 ;
int32_t gatesdim = d4*4;

auto inp = make_vector_float_rand(ALICE, num_samples, d2, d3) ;

// cout << ("Input inp:") << endl ;
// float *__tmp_in_inp = new float[1] ;
// for (uint32_t i0 = 0; i0 < num_samples; i0++){
// for (uint32_t i1 = 0; i1 < d2; i1++){
// for (uint32_t i2 = 0; i2 < d3; i2++){
// cin >> __tmp_in_inp[0];
// inp[i0][i1][i2] = __tmp_in_inp[0] ;
// }
// }
// }
// delete[] __tmp_in_inp ;


auto A1 = make_vector_float_rand(ALICE, d1, d2, d2) ;

// cout << ("Input A1:") << endl ;
// float *__tmp_in_A1 = new float[1] ;
// for (uint32_t i0 = 0; i0 < d1; i0++){
// for (uint32_t i1 = 0; i1 < d2; i1++){
// for (uint32_t i2 = 0; i2 < d2; i2++){
// cin >> __tmp_in_A1[0];
// A1[i0][i1][i2] = __tmp_in_A1[0] ;

// }
// }
// }
// delete[] __tmp_in_A1 ;

auto kernel1 = make_vector_float_rand(ALICE, d1, d3, d4) ;

// cout << ("Input kernel1:") << endl ;
// float *__tmp_in_kernel1 = new float[1] ;
// for (uint32_t i0 = 0; i0 < d1; i0++){
// for (uint32_t i1 = 0; i1 < d3; i1++){
// for (uint32_t i2 = 0; i2 < d4; i2++){
// cin >> __tmp_in_kernel1[0];
// kernel1[i0][i1][i2] = __tmp_in_kernel1[0] ;

// }
// }
// }
// delete[] __tmp_in_kernel1 ;

auto bias1 = make_vector_float_rand(ALICE, d2) ;

// cout << ("Input bias1:") << endl 
// float *__tmp_in_bias1 = new float[1] ;
// for (uint32_t i0 = 0; i0 < d2; i0++){
// cin >> __tmp_in_bias1[0];
// bias1[i0] = __tmp_in_bias1[0] ;

// }
// delete[] __tmp_in_bias1 ;

auto A2 = make_vector_float_rand(ALICE, d1, d2, d2) ;

// cout << ("Input A2:") << endl ;
// float *__tmp_in_A2 = new float[1] ;
// for (uint32_t i0 = 0; i0 < d1; i0++){
// for (uint32_t i1 = 0; i1 < d2; i1++){
// for (uint32_t i2 = 0; i2 < d2; i2++){
// cin >> __tmp_in_A2[0];
// A2[i0][i1][i2] = __tmp_in_A2[0] ;

// }
// }
// }
// delete[] __tmp_in_A2 ;

auto kernel2 = make_vector_float_rand(ALICE, d1, d4, d4) ;

// cout << ("Input kernel2:") << endl ;
// float *__tmp_in_kernel2 = new float[1] ;
// for (uint32_t i0 = 0; i0 < d1; i0++){
// for (uint32_t i1 = 0; i1 < d4; i1++){
// for (uint32_t i2 = 0; i2 < d4; i2++){
// cin >> __tmp_in_kernel2[0];
// kernel2[i0][i1][i2] = __tmp_in_kernel2[0] ;

// }
// }
// }
// delete[] __tmp_in_kernel2 ;

auto bias2 = make_vector_float_rand(ALICE, d2) ;

// cout << ("Input bias2:") << endl ;
// float *__tmp_in_bias2 = new float[1] ;
// for (uint32_t i0 = 0; i0 < d2; i0++){
// cin >> __tmp_in_bias2[0];
// bias2[i0] = __tmp_in_bias2[0] ;

// }
// delete[] __tmp_in_bias2 ;

auto hidden = make_vector_float_rand(ALICE, d1, d4) ;

// cout << ("Input hidden:") << endl ;
// float *__tmp_in_hidden = new float[1] ;
// for (uint32_t i0 = 0; i0 < d1; i0++){
// for (uint32_t i1 = 0; i1 < d4; i1++){
// cin >> __tmp_in_hidden[0];
// hidden[i0][i1] = __tmp_in_hidden[0] ;
// }
// }
// delete[] __tmp_in_hidden ;

auto cell = make_vector_float_rand(ALICE, d1, d4) ;

// cout << ("Input cell:") << endl ;
// float *__tmp_in_cell = new float[1] ;
// for (uint32_t i0 = 0; i0 < d1; i0++){
// for (uint32_t i1 = 0; i1 < d4; i1++){
// cin >> __tmp_in_cell[0];
// cell[i0][i1] = __tmp_in_cell[0] ;

// }
// }
// delete[] __tmp_in_cell ;

auto k = make_vector_float_rand(ALICE, d2, gatesdim) ;

// cout << ("Input k:") << endl ;
// float *__tmp_in_k = new float[1] ;
// for (uint32_t i0 = 0; i0 < d2; i0++){
// for (uint32_t i1 = 0; i1 < gatesdim; i1++){
// cin >> __tmp_in_k[0];
// k[i0][i1] = __tmp_in_k[0] ;
// }
// }

auto reck = make_vector_float_rand(ALICE, d4, gatesdim) ;

// cout << ("Input reck:") << endl ;
// float *__tmp_in_reck = new float[1] ;
// for (uint32_t i0 = 0; i0 < d4; i0++){
// for (uint32_t i1 = 0; i1 < gatesdim; i1++){
// cin >> __tmp_in_reck[0];
// reck[i0][i1] = __tmp_in_reck[0] ;
// }
// }

auto lstmbias = make_vector_float_rand(ALICE, gatesdim) ;

// cout << ("Input lstmbias:") << endl ;
// float *__tmp_in_lstmbias = new float[1] ;
// for (uint32_t i0 = 0; i0 < gatesdim; i0++){
// cin >> __tmp_in_lstmbias[0];
// lstmbias[i0] = __tmp_in_lstmbias[0] ;
// }

// delete[] __tmp_in_k ;
// delete[] __tmp_in_reck ;
// delete[] __tmp_in_lstmbias ;

auto dense = make_vector_float_rand(ALICE, d4, d2) ;

// cout << ("Input dense:") << endl ;
// float *__tmp_in_dense = new float[1] ;
// for (uint32_t i0 = 0; i0 < d4; i0++){
// for (uint32_t i1 = 0; i1 < d2; i1++){
// cin >> __tmp_in_dense[0];
// dense[i0][i1] = __tmp_in_dense[0] ;

// }
// }
// delete[] __tmp_in_dense ;

auto bias4 = make_vector_float_rand(ALICE, d2) ;

// cout << ("Input bias4:") << endl ;
// float *__tmp_in_bias4 = new float[1] ;
// for (uint32_t i0 = 0; i0 < d2; i0++){
// cin >> __tmp_in_bias4[0];
// bias4[i0] = __tmp_in_bias4[0] ;
// }

auto labs = make_vector_float_rand(ALICE, num_samples, d2) ;

// cout << ("Input labs:") << endl ;
// for (uint32_t i1 = 0; i1 < num_samples; i1++){
// for (uint32_t i0 = 0; i0 < d2; i0++){
// cin >> __tmp_in_bias4[0];
// labs[i1][i0] = __tmp_in_bias4[0] ;
// }
// }
// delete[] __tmp_in_bias4 ;

auto FullHt = make_vector_float_rand(ALICE, (d4 + 1), d1, d4) ;
auto FullIt = make_vector_float_rand(ALICE, d4, d1, d4) ;
auto FullFt = make_vector_float_rand(ALICE, d4, d1, d4) ;
auto FullGt = make_vector_float_rand(ALICE, d4, d1, d4) ;
auto FullOt = make_vector_float_rand(ALICE, d4, d1, d4) ;
auto FullCt = make_vector_float_rand(ALICE, (d4 + 1), d1, d4) ;
auto FullXt = make_vector_float_rand(ALICE, d4, d1, d2)  ;
auto neight2 = make_vector_float_rand(ALICE, d2, d4) ;
auto feat2 = make_vector_float_rand(ALICE, d2, d4) ;
auto neigh1 = make_vector_float_rand(ALICE, d3, d2) ;
auto feat1 = make_vector_float_rand(ALICE, d2, d3) ;
auto finaloutput = make_vector_float_rand(ALICE, d1, d2) ;
auto reluoutputs1 = make_vector_bool_rand(ALICE, d2, d4) ;
auto reluoutputs2 = make_vector_bool_rand(ALICE, d2, d4) ;
auto inputpoint = make_vector_float_rand(ALICE, d1, d2, d3) ;
auto labels = make_vector_float_rand(ALICE, d1, d2) ;


for(uint32_t iterations=0; iterations<20; iterations++)
{
    uint32_t ind = 0 ;
    AssignSampleFromData3D(d2, d3, ind, inp, inputpoint);
    AssignSampleFromData2D(d2, ind, labs, labels);

    forward(d1, d2, d3, d4, inputpoint, A1, kernel1, bias1, A2, kernel2, bias2, d4, d2, hdim, d1, gatesdim, hidden, cell, k, reck, lstmbias, dense, bias4, FullHt, FullIt, FullFt, FullGt, FullOt, FullCt, FullXt, neight2, feat2, neigh1, feat1, reluoutputs1, reluoutputs2, finaloutput);
    backward(d1, d2, d3, 16, 16, gatesdim, 16, dense, bias4, FullHt, FullIt, FullFt, FullGt, FullOt, FullCt, FullXt, finaloutput, labels, k, reck, lstmbias, neight2, kernel2, feat2, A2, neigh1, kernel1, feat1, A2, bias1, bias2, reluoutputs1, reluoutputs2);
}

return 0;
}