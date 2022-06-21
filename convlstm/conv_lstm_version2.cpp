
#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std ;
template<typename T>
vector<T> make_vector(size_t size) {
return std::vector<T>(size) ;
}

template <typename T, typename... Args>
auto make_vector(size_t first, Args... sizes)
{
auto inner = make_vector<T>(sizes...) ;
return vector<decltype(inner)>(first, inner) ;
}



void ClearMemSecret2(int32_t s1, int32_t s2, auto& arr){

/* Empty Function */

}

void MatMul2D(int32_t m, int32_t n, int32_t p, auto& A, auto& B, auto& mult, bool modelIsA){
for (uint32_t i = 0; i < m; i++){
for (uint32_t j = 0; j < p; j++){
mult[i][j] = 0. ;

}
}
for (uint32_t i = 0; i < m; i++){
for (uint32_t j = 0; j < p; j++){
for (uint32_t k = 0; k < n; k++){
float __tac_var1 = mult[i][j] ;

float __tac_var2 = A[i][k] ;

float __tac_var3 = (__tac_var1 + __tac_var2) ;

float __tac_var4 = B[k][j] ;

mult[i][j] = (__tac_var3 * __tac_var4) ;

}
}
}
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
auto inputReshaped = make_vector<float>(reshapedFilterCols, reshapedIPCols) ;

auto matmulOP = make_vector<float>(reshapedFilterRows, reshapedIPCols) ;

auto filterReshaped = make_vector<float>(reshapedFilterRows, reshapedFilterCols) ;

Conv2DReshapeFilterGroup(FH, FW, CI, CO, g, G, filterArr, filterReshaped);
Conv2DReshapeInputGroup(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, g, G, reshapedFilterCols, reshapedIPCols, inputArr, inputReshaped);
bool __tac_var37 = 1 ;

MatMul2D(reshapedFilterRows, reshapedFilterCols, reshapedIPCols, filterReshaped, inputReshaped, matmulOP, __tac_var37);
Conv2DReshapeMatMulOPGroup(N, outH, outW, CO, g, G, matmulOP, outArr);
ClearMemSecret2(reshapedFilterRows, reshapedFilterCols, filterReshaped);
ClearMemSecret2(reshapedFilterCols, reshapedIPCols, inputReshaped);
ClearMemSecret2(reshapedFilterRows, reshapedIPCols, matmulOP);
}
}

void GemmAdd(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& op, auto& bias){
for (uint32_t i1 = 0; i1 < s4; i1++){
for (uint32_t i2 = 0; i2 < s1; i2++){
for (uint32_t i3 = 0; i3 < s2; i3++){
for (uint32_t i4 = 0; i4 < s3; i4++){
float __tac_var38 = op[i2][i3][i4][i1] ;

float __tac_var39 = bias[i1] ;

op[i2][i3][i4][i1] = (__tac_var38 + __tac_var39) ;

}
}
}
}
}

void Hadamard(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& arr1, auto& arr2, auto& outArr){
    //Reimplement Hadamard product
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
float __tac_var40 = arr1[i1][i2][i3][i4] ;

float __tac_var41 = arr2[i1][i2][i3][i4] ;

outArr[i1][i2][i3][i4] =1;

//outArr[i1][i2][i3][i4] = (__tac_var40 * __tac_var41) ;

}
}
}
}
}

void Add(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& arr1, auto& arr2, auto& outArr){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
float __tac_var42 = arr1[i1][i2][i3][i4] ;

float __tac_var43 = arr2[i1][i2][i3][i4] ;

outArr[i1][i2][i3][i4] = (__tac_var42 + __tac_var43) ;

}
}
}
}
}



void Sigmoid4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& inArr, auto& outArr){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
float expneg = exp(-inArr[i1][i2][i3][i4]) ;

outArr[i1][i2][i3][i4] = 1/(1+expneg) ;

}
}
}
}
}



void Tanh4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& inArr, auto& outArr){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
//cout<<s1<<" "<<s2<<" "<<s3<<" "<<s4<<endl;
//cout<<i1<<" "<<i2<<" "<<i3<<" "<<i4<<endl;
//float exp= -inArr[i1][i2][i3][i4];
//float expneg = exp(-inArr[i1][i2][i3][i4]) ;
//float exp_ = exp(inArr[i1][i2][i3][i4]) ;

//outArr[i1][i2][i3][i4] = (exp_-expneg)/(exp_+expneg) ;
//Have to implement secure version anyway
}
}
}
}
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
auto outputarr = make_vector<float>(1, 270, 270, 32) ;

auto igate = make_vector<float>(1, 270, 270, 8) ;

auto fgate = make_vector<float>(1, 270, 270, 8) ;

auto ggate = make_vector<float>(1, 270, 270, 8) ;

auto ogate = make_vector<float>(1, 270, 270, 8) ;

auto outputsigmoid = make_vector<float>(1, 270, 270, 32) ;

auto outputg = make_vector<float>(1, 270, 270, 8) ;

auto add1 = make_vector<float>(1, 270, 270, 8) ;

auto add2 = make_vector<float>(1, 270, 270, 8) ;

auto cnextupdated = make_vector<float>(1, 270, 270, 8) ;
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
auto InputAssigned = make_vector<float>(1, 270, 270, 1) ;

auto HAssigned = make_vector<float>(1, 270, 270, 8) ;

auto CAssigned = make_vector<float>(1, 270, 270, 8) ;

auto OutputInner = make_vector<float>(sequnits, 1, 270, 270, 8) ;

auto WeightsAssigned = make_vector<float>(3, 3, 9, 32) ;

auto BiasAssigned = make_vector<float>(32) ;

auto InputHidden = make_vector<float>(1, 270, 270, 9) ;

for (uint32_t i = 0; i < numlayers; i++){
Assign4(3, 3, 9, 32, i, WeightsUnit, WeightsAssigned);
Assign4(1, 270, 270, 8, i, HUnit, HAssigned);
Assign4(1, 270, 270, 8, i, CUnit, CAssigned);
Assign2(32, i, BiasUnit, BiasAssigned);

for (uint32_t j = 0; j < sequnits; j++){
Assign4(1, 270, 270, 1, j, InputOriginal, InputAssigned);
ConcatInputHidden(1, 270, 270, 1, 8, InputAssigned, HAssigned, InputHidden);
cout<<"Seg fault here"<<endl;
ConvLSTMUnitCell(WeightsAssigned, InputHidden, BiasAssigned, CAssigned, HAssigned);
cout<<"exit"<<endl;
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

void forward(int32_t sequnits, int32_t numl, auto& Input, auto& Weights1, auto& Bias1, auto& H1, auto& C1, auto& Weights2, auto& Bias2, auto& Weights3, auto& Bias3){
auto LInputCells = make_vector<float>(numl, 1, 270, 270, 1) ;

auto LInputHidden = make_vector<float>(numl, 1, 270, 270, 8) ;

auto LastLayerOP = make_vector<float>(1, 270, 270, 8) ;

auto LInputCells2 = make_vector<float>(numl, 1, 270, 270, 1) ;

auto LInputHidden2 = make_vector<float>(numl, 1, 270, 270, 8) ;

auto outputarr = make_vector<float>(1, 270, 270, 1) ;
cout<<"Hello!"<<endl;
ConvLSTM(sequnits, numl, Input, Weights1, Bias1, H1, C1, LInputCells, LInputHidden, LastLayerOP);
cout<<"Hello!2"<<endl;
InitInputZero(sequnits, 1, 270, 270, 1, Input);
cout<<"Hello!3"<<endl;
ConvLSTM(1, numl, Input, Weights2, Bias2, LInputHidden, LInputCells, LInputCells2, LInputHidden2, LastLayerOP);
Conv2DGroup(1, 270, 270, 8, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, LastLayerOP, Weights3, outputarr);
GemmAdd(1, 270, 270, 1, outputarr, Bias3);
}


int main (int __argc, char __argv) {
//__init(__argc, __argv);

auto inp = make_vector<float>(1, 1, 270, 270, 1) ;
auto Ct1 = make_vector<float>(1, 1, 270, 270, 8) ;
auto Ht1 = make_vector<float>(1, 1, 270, 270, 8) ;
auto W1 = make_vector<float>(1, 3, 3, 9, 32) ;
auto b2 = make_vector<float>(1, 32) ;
auto b1 = make_vector<float>(1, 32) ;
auto W2 = make_vector<float>(1, 3, 3, 9, 32) ;
auto W3 = make_vector<float>(1, 1, 8, 1) ;
auto b3 = make_vector<float>(1) ;


/*
cout << ("Input inp:") << endl ;

float *__tmp_in_inp = new float[1] ;

for (uint32_t i0 = 0; i0 < 1; i0++){
for (uint32_t i1 = 0; i1 < 1; i1++){
for (uint32_t i2 = 0; i2 < 5; i2++){
for (uint32_t i3 = 0; i3 < 5; i3++){
for (uint32_t i4 = 0; i4 < 1; i4++){
cin >> __tmp_in_inp[0];
inp[i0][i1][i2][i3][i4] = __tmp_in_inp[0] ;

}
}
}
}
}
delete[] __tmp_in_inp ;

auto Ct1 = make_vector<float>(1, 1, 5, 5, 5) ;

cout << ("Input Ct1:") << endl ;

float *__tmp_in_Ct1 = new float[1] ;

for (uint32_t i0 = 0; i0 < 1; i0++){
for (uint32_t i1 = 0; i1 < 1; i1++){
for (uint32_t i2 = 0; i2 < 5; i2++){
for (uint32_t i3 = 0; i3 < 5; i3++){
for (uint32_t i4 = 0; i4 < 5; i4++){
cin >> __tmp_in_Ct1[0];
Ct1[i0][i1][i2][i3][i4] = __tmp_in_Ct1[0] ;

}
}
}
}
}
delete[] __tmp_in_Ct1 ;

auto Ct1 = make_vector<float>(1, 1, 5, 5, 5) ;

cout << ("Input Ht1:") << endl ;

float *__tmp_in_Ht1 = new float[1] ;

for (uint32_t i0 = 0; i0 < 1; i0++){
for (uint32_t i1 = 0; i1 < 1; i1++){
for (uint32_t i2 = 0; i2 < 5; i2++){
for (uint32_t i3 = 0; i3 < 5; i3++){
for (uint32_t i4 = 0; i4 < 5; i4++){
cin >> __tmp_in_Ht1[0];
Ht1[i0][i1][i2][i3][i4] = __tmp_in_Ht1[0] ;

}
}
}
}
}
delete[] __tmp_in_Ht1 ;

auto W1 = make_vector<float>(1, 3, 3, 6, 20) ;

cout << ("Input W1:") << endl ;

float *__tmp_in_W1 = new float[1] ;

for (uint32_t i0 = 0; i0 < 1; i0++){
for (uint32_t i4 = 0; i4 < 20; i4++){
for (uint32_t i3 = 0; i3 < 6; i3++){
for (uint32_t i1 = 0; i1 < 3; i1++){
for (uint32_t i2 = 0; i2 < 3; i2++){
cin >> __tmp_in_W1[0];
W1[i0][i1][i2][i3][i4] = __tmp_in_W1[0] ;

}
}
}
}
}
delete[] __tmp_in_W1 ;

auto b1 = make_vector<float>(1, 20) ;

cout << ("Input b1:") << endl ;

float *__tmp_in_b1 = new float[1] ;

for (uint32_t i0 = 0; i0 < 1; i0++){
for (uint32_t i1 = 0; i1 < 20; i1++){
cin >> __tmp_in_b1[0];
b1[i0][i1] = __tmp_in_b1[0] ;

}
}
delete[] __tmp_in_b1 ;

auto W2 = make_vector<float>(1, 3, 3, 6, 20) ;

cout << ("Input W2:") << endl ;

float *__tmp_in_W2 = new float[1] ;

for (uint32_t i0 = 0; i0 < 1; i0++){
for (uint32_t i4 = 0; i4 < 20; i4++){
for (uint32_t i3 = 0; i3 < 6; i3++){
for (uint32_t i1 = 0; i1 < 3; i1++){
for (uint32_t i2 = 0; i2 < 3; i2++){
cin >> __tmp_in_W2[0];
W2[i0][i1][i2][i3][i4] = __tmp_in_W2[0] ;

}
}
}
}
}
delete[] __tmp_in_W2 ;

auto b2 = make_vector<float>(1, 20) ;

cout << ("Input b2:") << endl ;

float *__tmp_in_b2 = new float[1] ;

for (uint32_t i0 = 0; i0 < 1; i0++){
for (uint32_t i1 = 0; i1 < 20; i1++){
cin >> __tmp_in_b2[0];
b2[i0][i1] = __tmp_in_b2[0] ;

}
}
delete[] __tmp_in_b2 ;

auto W3 = make_vector<float>(1, 1, 5, 1) ;

cout << ("Input W3:") << endl ;

float *__tmp_in_W3 = new float[1] ;

for (uint32_t i0 = 0; i0 < 1; i0++){
for (uint32_t i1 = 0; i1 < 1; i1++){
for (uint32_t i2 = 0; i2 < 5; i2++){
for (uint32_t i3 = 0; i3 < 1; i3++){
cin >> __tmp_in_W3[0];
W3[i0][i1][i2][i3] = __tmp_in_W3[0] ;

}
}
}
}
delete[] __tmp_in_W3 ;

auto b3 = make_vector<float>(1) ;

cout << ("Input b3:") << endl ;

float *__tmp_in_b3 = new float[1] ;

for (uint32_t i0 = 0; i0 < 1; i0++){
cin >> __tmp_in_b3[0];
b3[i0] = __tmp_in_b3[0] ;

}
delete[] __tmp_in_b3 ;

int32_t iters = 1 ;
*/
for (uint32_t i = 0; i < 1; i++){
forward(1, 1, inp, W1, b1, Ht1, Ct1, W2, b2, W3, b3);
}
return 0;
}

