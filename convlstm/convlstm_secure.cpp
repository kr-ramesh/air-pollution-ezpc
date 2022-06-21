
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
extern void GemmAdd3(int32_t s1, int32_t s2, int32_t s3, vector<vector<vector<FPArray>>> &inArr, vector<FPArray> &bias, vector<vector<vector<FPArray>>> &outArr) ;
extern void dotProduct2(int32_t s1, int32_t s2, vector < vector < FPArray > >& arr1, vector < vector < FPArray > >& arr2, vector < FPArray >& outArr);
extern void Relu(int32_t s1, vector < FPArray >& inArr, vector < FPArray >& outArr, vector < BoolArray >& hotArr);
extern void getBiasDer(int32_t s1, int32_t s2, vector < vector < FPArray > >& der, vector < FPArray >& biasDer);
extern void IfElse(int32_t s1, vector < FPArray >& dat, vector < BoolArray >& hot, vector < FPArray >& out, bool flip);
extern void updateWeights(int32_t s, float lr, vector < FPArray >& bias, vector < FPArray >& der);
extern void updateWeightsAdam(int32_t s1, int32_t t, float lr, float beta1, float beta2, float eps, vector<FPArray>& inArr, vector<FPArray>& derArr, vector<FPArray>& mArr, vector<FPArray>& vArr) ;
extern void getLoss(int32_t m, vector < FPArray >& lossTerms, vector < FPArray >& loss);

//Define GemmAdd4

//Not sure if the Conv2D functions are legit, find secure versions

void ClearMemSecret2(int32_t s1, int32_t s2, auto& arr){

/* Empty Function */

}

void Conv2DReshapeMatMulOPGroup(int32_t N, int32_t finalH, int32_t finalW, int32_t CO, int32_t g, int32_t G, auto& inputArr, auto& outputArr){
int32_t COG = (CO / G) ;

int32_t startCO = (g * COG) ;

for (uint32_t co = 0; co < COG; co++){
for (uint32_t n = 0; n < N; n++){
for (uint32_t h = 0; h < finalH; h++){
for (uint32_t w = 0; w < finalW; w++){
int32_t __tac_var5 = (co + startCO) ;

int32_t __tac_var6 = (n * finalH) ;

int32_t __tac_var7 = (__tac_var6 * finalW) ;

int32_t __tac_var8 = (h * finalW) ;

int32_t __tac_var9 = (__tac_var7 + __tac_var8) ;

int32_t __tac_var10 = (__tac_var9 + w) ;

outputArr[n][h][w][__tac_var5] = inputArr[co][__tac_var10] ;

}
}
}
}
}

void Conv2DReshapeInputGroup(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t g, int32_t G, int32_t RRows, int32_t RCols, auto& inputArr, auto& outputArr){
int32_t linIdxFilterMult = 0 ;

int32_t CIG = (CI / G) ;

for (uint32_t n = 0; n < N; n++){
int32_t leftTopCornerH = (0 - zPadHLeft) ;

int32_t __tac_var11 = (H - 1) ;

int32_t extremeRightBottomCornerH = (__tac_var11 + zPadHRight) ;

while ((((leftTopCornerH + FH) - 1) <= extremeRightBottomCornerH)) {
int32_t leftTopCornerW = (0 - zPadWLeft) ;

int32_t __tac_var12 = (W - 1) ;

int32_t extremeRightBottomCornerW = (__tac_var12 + zPadWRight) ;

while ((((leftTopCornerW + FW) - 1) <= extremeRightBottomCornerW)) {
for (uint32_t fh = 0; fh < FH; fh++){
for (uint32_t fw = 0; fw < FW; fw++){
int32_t curPosH = (leftTopCornerH + fh) ;

int32_t curPosW = (leftTopCornerW + fw) ;

float val = 0. ;

int32_t startCI = (g * CIG) ;

for (uint32_t ci = 0; ci < CIG; ci++){
bool __tac_var13 = (curPosH < 0) ;

bool __tac_var14 = (curPosH >= H) ;

bool __tac_var15 = (__tac_var13 || __tac_var14) ;

bool __tac_var16 = (curPosW < 0) ;

bool __tac_var17 = (curPosW >= W) ;

bool __tac_var18 = (__tac_var16 || __tac_var17) ;

bool __tac_var19 = (__tac_var15 || __tac_var18) ;

if (__tac_var19) {
val = 0. ;

} else {
int32_t __tac_var20 = (ci + startCI) ;

val = inputArr[n][curPosH][curPosW][__tac_var20] ;

}
outputArr[((((fh * FW) * CIG) + (fw * CIG)) + ci)][linIdxFilterMult] = val ;

}
}
}
linIdxFilterMult = (linIdxFilterMult + 1) ;

leftTopCornerW = (leftTopCornerW + strideW) ;

}

leftTopCornerH = (leftTopCornerH + strideH) ;

}

}
}

void Conv2DReshapeFilterGroup(int32_t FH, int32_t FW, int32_t CI, int32_t CO, int32_t g, int32_t G, auto& inputArr, auto& outputArr){
int32_t CIG = (CI / G) ;

int32_t COG = (CO / G) ;

int32_t startCO = (g * COG) ;

for (uint32_t co = 0; co < COG; co++){
for (uint32_t fh = 0; fh < FH; fh++){
for (uint32_t fw = 0; fw < FW; fw++){
for (uint32_t ci = 0; ci < CIG; ci++){
int32_t __tac_var21 = (fh * FW) ;

int32_t __tac_var22 = (__tac_var21 * CIG) ;

int32_t __tac_var23 = (fw * CIG) ;

int32_t __tac_var24 = (__tac_var22 + __tac_var23) ;

int32_t linIdx = (__tac_var24 + ci) ;

int32_t __tac_var25 = (co + startCO) ;

outputArr[co][linIdx] = inputArr[fh][fw][ci][__tac_var25] ;

}
}
}
}
}

void Conv2DGroup(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t G, auto& inputArr, auto& filterArr, auto& outArr){
int32_t CIG = (CI / G) ;

int32_t reshapedFilterRows = (CO / G) ;

int32_t __tac_var26 = (FH * FW) ;

int32_t reshapedFilterCols = (__tac_var26 * CIG) ;

int32_t __tac_var27 = __tac_var26 ;

int32_t reshapedIPRows = reshapedFilterCols ;

int32_t __tac_var28 = (zPadHLeft + zPadHRight) ;

int32_t __tac_var29 = (H + __tac_var28) ;

int32_t __tac_var30 = (__tac_var29 - FH) ;

int32_t __tac_var31 = (__tac_var30 / strideH) ;

int32_t outH = (__tac_var31 + 1) ;

int32_t __tac_var32 = (zPadWLeft + zPadWRight) ;

int32_t __tac_var33 = (W + __tac_var32) ;

int32_t __tac_var34 = (__tac_var33 - FW) ;

int32_t __tac_var35 = (__tac_var34 / strideW) ;

int32_t outW = (__tac_var35 + 1) ;

int32_t __tac_var36 = (N * outH) ;

int32_t reshapedIPCols = (__tac_var36 * outW) ;

for (uint32_t g = 0; g < G; g++){
auto inputReshaped = make_vector_float(ALICE, reshapedFilterCols, reshapedIPCols) ;

auto matmulOP = make_vector_float(ALICE, reshapedFilterRows, reshapedIPCols) ;

auto filterReshaped = make_vector_float(ALICE, reshapedFilterRows, reshapedFilterCols) ;

Conv2DReshapeFilterGroup(FH, FW, CI, CO, g, G, filterArr, filterReshaped);
Conv2DReshapeInputGroup(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, g, G, reshapedFilterCols, reshapedIPCols, inputArr, inputReshaped);
bool __tac_var37 = 1 ;

MatMul(reshapedFilterRows, reshapedFilterCols, reshapedIPCols, filterReshaped, inputReshaped, matmulOP, __tac_var37);
Conv2DReshapeMatMulOPGroup(N, outH, outW, CO, g, G, matmulOP, outArr);
ClearMemSecret2(reshapedFilterRows, reshapedFilterCols, filterReshaped);
ClearMemSecret2(reshapedFilterCols, reshapedIPCols, inputReshaped);
ClearMemSecret2(reshapedFilterRows, reshapedIPCols, matmulOP);
}
}

void ElemWiseMul4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector <vector <vector < vector < FPArray > > > >& arr1, vector <vector <vector < vector < FPArray > > > >& arr2, vector <vector <vector < vector < FPArray > > > >& outArr) {
    int sz = s1*s2*s3*s4 ;

    vector<FPArray> arr1_flat = make_vector_float(ALICE, sz) ;
    vector<FPArray> arr2_flat = make_vector_float(ALICE, sz) ;
    vector<FPArray> outarr_flat = make_vector_float(ALICE, sz) ;

    for (uint32_t i1 = 0; i1 < s1; i1++){
    for (uint32_t i2 = 0; i2 < s2; i2++){
    for (uint32_t i3 = 0; i3 < s3; i3++){
    for (uint32_t i4 = 0; i4 < s4; i4++){
            arr1_flat[(i1 * s2) + (i2* s3) + (i3 * s4) + i4] = arr1[i1][i2][i3][i4] ;
            arr2_flat[(i1 * s2) + (i2* s3) + (i3 * s4) + i4] = arr2[i1][i2][i3][i4] ;
    } } } }

    ElemWiseMul(sz, arr1_flat, arr2_flat, outarr_flat) ;
    
    for (uint32_t i1 = 0; i1 < s1; i1++){
    for (uint32_t i2 = 0; i2 < s2; i2++){
    for (uint32_t i3 = 0; i3 < s3; i3++){
    for (uint32_t i4 = 0; i4 < s4; i4++){
            outArr[i1][i2][i3][i4] = outarr_flat[(i1 * s2) + (i2* s3) + (i3 * s4) + i4] ;
    } } } }
}

void ElemWiseAdd4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector <vector <vector < vector < FPArray > > > >& arr1, vector <vector <vector < vector < FPArray > > > >& arr2, vector <vector <vector < vector < FPArray > > > >& outArr) {
    int sz = s1*s2*s3*s4 ;

    vector<FPArray> arr1_flat = make_vector_float(ALICE, sz) ;
    vector<FPArray> arr2_flat = make_vector_float(ALICE, sz) ;
    vector<FPArray> outarr_flat = make_vector_float(ALICE, sz) ;

    for (uint32_t i1 = 0; i1 < s1; i1++){
    for (uint32_t i2 = 0; i2 < s2; i2++){
    for (uint32_t i3 = 0; i3 < s3; i3++){
    for (uint32_t i4 = 0; i4 < s4; i4++){
            arr1_flat[(i1 * s2) + (i2* s3) + (i3 * s4) + i4] = arr1[i1][i2][i3][i4] ;
            arr2_flat[(i1 * s2) + (i2* s3) + (i3 * s4) + i4] = arr2[i1][i2][i3][i4] ;
    } } } }

    ElemWiseAdd(sz, arr1_flat, arr2_flat, outarr_flat) ;

    for (uint32_t i1 = 0; i1 < s1; i1++){
    for (uint32_t i2 = 0; i2 < s2; i2++){
    for (uint32_t i3 = 0; i3 < s3; i3++){
    for (uint32_t i4 = 0; i4 < s4; i4++){
            outArr[i1][i2][i3][i4] = outarr_flat[(i1 * s2) + (i2* s3) + (i3 * s4) + i4] ;
    } } } }
}

void Sigmoid4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector <vector <vector < vector < FPArray > > > >& inArr, vector <vector <vector < vector < FPArray > > > >& outArr){
int32_t sz = (s1 * s2 * s3* s4) ;

vector < FPArray > inFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > outFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
inFlat[(i1 * s2) + (i2* s3) + (i3 * s4) + i4] = inArr[i1][i2][i3][i4] ;
} } } }

Sigmoid(sz, inFlat, outFlat);
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
outArr[i1][i2][i3][i4] = outFlat[(i1 * s2) + (i2* s3) + (i3 * s4) + i4] ;
} } } }

}

void Tanh4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, vector <vector <vector < vector < FPArray > > > >& inArr, vector <vector <vector < vector < FPArray > > > >& outArr){
int32_t sz = (s1 * s2 * s3* s4) ;

vector < FPArray > inFlat = make_vector_float(ALICE, sz) ;

vector < FPArray > outFlat = make_vector_float(ALICE, sz) ;

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
inFlat[(i1 * s2) + (i2* s3) + (i3 * s4) + i4] = inArr[i1][i2][i3][i4] ;
} } } }

Tanh(sz, inFlat, outFlat);

for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
outArr[i1][i2][i3][i4] = outFlat[(i1 * s2) + (i2* s3) + (i3 * s4) + i4] ;
} } } }
}



void InitHiddenStates(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& Ht){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
Ht[i1][i2][i3][i4] = 0. ;

}
}
}
}
}

void InitInputZero(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t s5, auto& I){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
for (uint32_t i5 = 0; i5 < s5; i5++){
I[i1][i2][i3][i4][i5] = 0. ;

}
}
}
}
}
}

void ConvLSTMUnitCell(auto& layer1W, auto& XH, auto& b, auto& Ct1, auto& Ht){
auto outputarr = make_vector_float(ALICE, 1, 270, 270, 32) ;
auto igate = make_vector_float(ALICE, 1, 270, 270, 8) ;
auto fgate = make_vector_float(ALICE, 1, 270, 270, 8) ;
auto ggate = make_vector_float(ALICE, 1, 270, 270, 8) ;
auto ogate = make_vector_float(ALICE, 1, 270, 270, 8) ;
auto outputsigmoid = make_vector_float(ALICE, 1, 270, 270, 32) ;
auto outputg = make_vector_float(ALICE, 1, 270, 270, 8) ;
auto add1 = make_vector_float(ALICE, 1, 270, 270, 8) ;
auto add2 = make_vector_float(ALICE, 1, 270, 270, 8) ;
auto cnextupdated = make_vector_float(ALICE, 1, 270, 270, 8) ;

Conv2DGroup(1, 270, 270, 9, 3, 3, 32, 1, 1, 1, 1, 1, 1, 1, XH, layer1W, outputarr);
GemmAdd(1, 270, 270, 32, outputarr, b);
Sigmoid4(1, 270, 270, 32, outputarr, outputsigmoid);
for (uint32_t l = 0; l < 1; l++){
for (uint32_t j = 0; j < 270; j++){
for (uint32_t k = 0; k < 270; k++){
for (uint32_t i = 0; i < 8; i++){
igate[l][i][j][k] = outputsigmoid[l][j][k][i] ;

int32_t __tac_var44 = (5 + i) ;

fgate[l][i][j][k] = outputsigmoid[l][j][k][__tac_var44] ;

int32_t __tac_var45 = (10 + i) ;

ogate[l][i][j][k] = outputsigmoid[l][j][k][__tac_var45] ;

int32_t __tac_var46 = (15 + i) ;

ggate[l][i][j][k] = outputarr[l][j][k][__tac_var46] ;

}
}
}
}
Tanh4(1, 270, 270, 8, ggate, outputg);
Hadamard(1, 270, 270, 8, fgate, Ct1, add1);
Hadamard(1, 270, 270, 8, igate, outputg, add2);
Add(1, 270, 270, 8, add1, add2, Ct1);
Tanh4(1, 270, 270, 8, Ct1, cnextupdated);
Hadamard(1, 270, 270, 8, ogate, cnextupdated, Ht);
}


void Assign4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t s5, auto& arr1, auto& arr2){
for (uint32_t l = 0; l < s1; l++){
for (uint32_t j = 0; j < s2; j++){
for (uint32_t k = 0; k < s3; k++){
for (uint32_t i = 0; i < s4; i++){
arr2[l][j][k][i] = arr1[s5][l][j][k][i] ;

}
}
}
}
}

void Reassign4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t s5, auto& arr2, auto& arr1){
for (uint32_t l = 0; l < s1; l++){
for (uint32_t j = 0; j < s2; j++){
for (uint32_t k = 0; k < s3; k++){
for (uint32_t i = 0; i < s4; i++){
arr2[s5][l][j][k][i] = arr1[l][j][k][i] ;

}
}
}
}
}

void Transfer(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t s5, auto& arr2, auto& arr1){
for (uint32_t l = 0; l < s1; l++){
for (uint32_t j = 0; j < s2; j++){
for (uint32_t k = 0; k < s3; k++){
for (uint32_t i = 0; i < s4; i++){
for (uint32_t m = 0; m < s5; m++){
arr2[l][j][k][i][m] = arr1[l][j][k][i][m] ;

}
}
}
}
}
}

void Assign2(int32_t s1, int32_t s2, auto& arr1, auto& arr2){
for (uint32_t l = 0; l < s1; l++){
arr2[l] = arr1[s2][l] ;
}
}

void ConcatInputHidden(int32_t s1, int32_t s2, int32_t s3, int32_t s4, int32_t s5, auto& I, auto& H, auto& O){
for (uint32_t l = 0; l < s1; l++){
for (uint32_t j = 0; j < s2; j++){
for (uint32_t k = 0; k < s3; k++){
for (uint32_t i = 0; i < s4; i++){
O[l][j][k][i] = I[l][j][k][i] ;

}
}
}
}
for (uint32_t l = 0; l < s1; l++){
for (uint32_t j = 0; j < s2; j++){
for (uint32_t k = 0; k < s3; k++){
for (uint32_t i = 0; i < s5; i++){
int32_t __tac_var47 = (s4 + i) ;

O[l][j][k][__tac_var47] = H[l][j][k][i] ;

}
}
}
}
}

void ConvLSTM(int32_t sequnits, int32_t numlayers, auto& InputOriginal, auto& WeightsUnit, auto& BiasUnit, auto& HUnit, auto& CUnit, auto& LastInputCells, auto& LastInputHidden, auto& LayerOutputs){
auto InputAssigned = make_vector_float(ALICE, 1, 270, 270, 1) ;

auto HAssigned = make_vector_float(ALICE,1, 270, 270, 8) ;

auto CAssigned = make_vector_float(ALICE,1, 270, 270, 8) ;

auto OutputInner = make_vector_float(ALICE,sequnits, 1, 270, 270, 8) ;

auto WeightsAssigned = make_vector_float(ALICE, 3, 3, 9, 32) ;

auto BiasAssigned = make_vector_float(ALICE, 32) ;

auto InputHidden = make_vector_float(ALICE, 1, 270, 270, 9) ;

for (uint32_t i = 0; i < numlayers; i++){
Assign4(3, 3, 9, 32, i, WeightsUnit, WeightsAssigned);
Assign4(1, 270, 270, 8, i, HUnit, HAssigned);
Assign4(1, 270, 270, 8, i, CUnit, CAssigned);
Assign2(32, i, BiasUnit, BiasAssigned);

for (uint32_t j = 0; j < sequnits; j++){
Assign4(1, 270, 270, 1, j, InputOriginal, InputAssigned);
ConcatInputHidden(1, 270, 270, 1, 8, InputAssigned, HAssigned, InputHidden);
ConvLSTMUnitCell(WeightsAssigned, InputHidden, BiasAssigned, CAssigned, HAssigned);
Reassign4(1, 270, 270, 8, j, OutputInner, HAssigned);
}
for (uint32_t j = 0; j < sequnits; j++)
    Transfer(1, 270, 270, 8, j, InputOriginal, OutputInner);

Reassign4(1, 270, 270, 8, i, LastInputHidden, HAssigned);
Reassign4(1, 270, 270, 8, i, LastInputCells, CAssigned);
}
int32_t __tac_var48 = (sequnits - 1) ;
Assign4(1, 270, 270, 8, __tac_var48, OutputInner, LayerOutputs);
}


void forward(int32_t sequnits, int32_t numl, auto& Input, auto& Weights1, auto& Bias1, auto& H1, auto& C1, auto& Weights2, auto& Bias2, auto& Weights3, auto& Bias3)
{
auto LInputCells = make_vector_float(ALICE, numl, 1, 270, 270, 1) ;

auto LInputHidden = make_vector_float(ALICE, numl, 1, 270, 270, 8) ;

auto LastLayerOP = make_vector_float(ALICE, 1, 270, 270, 8) ;

auto LInputCells2 = make_vector_float(ALICE, numl, 1, 270, 270, 1) ;

auto LInputHidden2 = make_vector_float(ALICE, numl, 1, 270, 270, 8) ;

auto outputarr = make_vector_float(ALICE, 1, 270, 270, 1) ;
cout<<"Hello!"<<endl;
ConvLSTM(sequnits, numl, Input, Weights1, Bias1, H1, C1, LInputCells, LInputHidden, LastLayerOP);
cout<<"Hello!2"<<endl;
InitInputZero(sequnits, 1, 270, 270, 1, Input);
cout<<"Hello!3"<<endl;
ConvLSTM(1, numl, Input, Weights2, Bias2, LInputHidden, LInputCells, LInputCells2, LInputHidden2, LastLayerOP);
Conv2DGroup(1, 270, 270, 8, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, LastLayerOP, Weights3, outputarr);
GemmAdd(1, 270, 270, 1, outputarr, Bias3);
}

int main (int __argc, char **__argv) {
__init(__argc, __argv);

 /* ArgMapping __amap;
  	__amap.arg("r", __party, "Role of party: ALICE/SERVER = 1; BOB/CLIENT = 2") ;
	__amap.arg("mbits", __m_bits, "mantissa bits") ;
	__amap.arg("ebits", __e_bits, "exponent bits") ;
	__amap.arg("port", port, "port") ;
	__amap.arg("add", address, "address") ;
  __amap.parse(__argc, __argv)*/

int32_t num_samples=33;
int32_t d1 = 23;
int32_t d2 = 270 ;
int32_t d3 = 3; 
int32_t d4 = 4 ;
int32_t hdim = 4;
int32_t gatesdim = d4*4;

auto inp = make_vector_float(ALICE, 1, 1, 270, 270, 1) ;
auto Ct1 = make_vector_float(ALICE, 1, 1, 270, 270, 8) ;
auto Ht1 = make_vector_float(ALICE, 1, 1, 270, 270, 8) ;
auto W1 = make_vector_float(ALICE, 1, 3, 3, 9, 32) ;
auto b2 = make_vector_float(ALICE, 1, 32) ;
auto b1 = make_vector_float(ALICE, 1, 32) ;
auto W2 = make_vector_float(ALICE, 1, 3, 3, 9, 32) ;
auto W3 = make_vector_float(ALICE, 1, 1, 8, 1) ;
auto b3 = make_vector_float(ALICE, 1) ; 

auto start = clock_start();

for (uint32_t i = 0; i < 1; i++){
forward(1, 1, inp, W1, b1, Ht1, Ct1, W2, b2, W3, b3);
}

long long t = time_from(start);
cout << "Total Time:\t" << t / (1000.0) << " ms" << endl;

return 0;
}