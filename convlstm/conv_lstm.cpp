#include <vector>
#include <math.h>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "FloatingPoint/floating-point.h"
#include "FloatingPoint/fp-math.h"
#include "secfloat.h"

using namespace std ;
using namespace sci ;


void ClearMemSecret1(int32_t s1, auto& arr){
return  ;

}

void ClearMemSecret2(int32_t s1, int32_t s2, auto& arr){
return  ;

}

void ClearMemSecret3(int32_t s1, int32_t s2, int32_t s3, auto& arr){
return  ;

}

void ClearMemSecret4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& arr){
return  ;

}

void MatMul2D(
	int32_t m, 
	int32_t n, 
	int32_t p, 
	vector<vector<FPArray>> &A, 
	vector<vector<FPArray>> &B, 
	vector<vector<FPArray>> &C, 
	bool modelIsA) {
	// computes matrix product A x B
	// A is [m, n] and B is [n, p]

	int m_bits = A[0][0].m_bits ;
	int e_bits = A[0][0].e_bits ;

	vector<FPArray> rows ;
	uint8_t *rows_s = new uint8_t[n] ;
	uint8_t *rows_z = new uint8_t[n] ;
	uint64_t *rows_m = new uint64_t[n] ;
	uint64_t *rows_e = new uint64_t[n] ;
	for (int i = 0 ; i < m ; i++) {
		for (int j = 0 ; j < n ; j++) {
			rows_s[j] = A[i][j].s[0] ;
			rows_z[j] = A[i][j].z[0] ;
			rows_m[j] = A[i][j].m[0] ;
			rows_e[j] = A[i][j].e[0] ;
		}
		FPArray row = __fp_op->input(__party, n, rows_s, rows_z, rows_m, rows_e, m_bits, e_bits) ;
		rows.push_back(row) ;
	}

	vector<FPArray> cols ;
	uint8_t *cols_s = new uint8_t[n] ;
	uint8_t *cols_z = new uint8_t[n] ;
	uint64_t *cols_m = new uint64_t[n] ;
	uint64_t *cols_e = new uint64_t[n] ;
	for (int j = 0 ; j < p ; j++) {
		for (int i = 0 ; i < n ; i++) {
			cols_s[i] = B[i][j].s[0] ;
			cols_z[i] = B[i][j].z[0] ;
			cols_m[i] = B[i][j].m[0] ;
			cols_e[i] = B[i][j].e[0] ;
		}
		FPArray col = __fp_op->input(__party, n, cols_s, cols_z, cols_m, cols_e, m_bits, e_bits) ;
		cols.push_back(col) ;
	}

	vector<FPArray> dotx, doty ;
	for (int i = 0 ; i < m ; i++) {
		for (int j = 0 ; j < p ; j++) {
			dotx.push_back(rows[i]) ;
			doty.push_back(cols[j]) ;
		}
	}

	FPArray dp = __fp_op->dot_product(dotx, doty) ;
	int k = 0 ;
	for (int i = 0 ; i < m ; i++) {
		for (int j = 0 ; j < p ; j++) {
			C[i][j].s[0] = dp.s[k] ;
			C[i][j].z[0] = dp.z[k] ;
			C[i][j].m[0] = dp.m[k] ;
			C[i][j].e[0] = dp.e[k] ;
			// C[i][j] = dp.subset(k, k+1) ;
			k++ ;			
		}
	}
}


void ElemWiseSecretSharedVectorMult(int32_t s1, auto& arr1, auto& arr2, auto& outArr){
for (uint32_t ii = 0; ii < s1; ii++){
outArr[ii] = __fp_op->mul(arr1[ii], arr2[ii]) ;

}
}

void ElemWiseActModelVectorMult(int32_t s1, auto& arr1, auto& arr2, auto& outArr){
ElemWiseSecretSharedVectorMult(s1, arr1, arr2, outArr);
}


void ElemWiseDiv(int32_t s1, auto& arr1, auto& arr2, auto& outArr){
for (uint32_t ii = 0; ii < s1; ii++){
outArr[ii] = __fp_op->div(arr1[ii], arr2[ii]) ;

}
}

void Conv2DReshapeMatMulOPGroup(int32_t N, int32_t finalH, int32_t finalW, int32_t CO, int32_t g, int32_t G, auto& inputArr, auto& outputArr){
int32_t COG = (CO / G) ;

int32_t startCO = (g * COG) ;

for (uint32_t co = 0; co < COG; co++){
for (uint32_t n = 0; n < N; n++){
for (uint32_t h = 0; h < finalH; h++){
for (uint32_t w = 0; w < finalW; w++){
outputArr[n][h][w][(co + startCO)] = inputArr[co][((((n * finalH) * finalW) + (h * finalW)) + w)] ;

}
}
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
int32_t linIdx = ((((fh * FW) * CIG) + (fw * CIG)) + ci) ;

outputArr[co][linIdx] = inputArr[fh][fw][ci][(co + startCO)] ;

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

int32_t extremeRightBottomCornerH = ((H - 1) + zPadHRight) ;

while ((((leftTopCornerH + FH) - 1) <= extremeRightBottomCornerH)) {
int32_t leftTopCornerW = (0 - zPadWLeft) ;

int32_t extremeRightBottomCornerW = ((W - 1) + zPadWRight) ;

while ((((leftTopCornerW + FW) - 1) <= extremeRightBottomCornerW)) {
for (uint32_t fh = 0; fh < FH; fh++){
for (uint32_t fw = 0; fw < FW; fw++){
int32_t curPosH = (leftTopCornerH + fh) ;

int32_t curPosW = (leftTopCornerW + fw) ;

FPArray val = __public_float_to_arithmetic(0.) ;

int32_t startCI = (g * CIG) ;

for (uint32_t ci = 0; ci < CIG; ci++){
if ((((curPosH < 0) || (curPosH >= H)) || ((curPosW < 0) || (curPosW >= W)))) {
val = __public_float_to_arithmetic(0.) ;

} else {
val = inputArr[n][curPosH][curPosW][(ci + startCI)] ;

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



void Conv2DGroup(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t G, auto& inputArr, auto& filterArr, auto& outArr){
int32_t CIG = (CI / G) ;

int32_t reshapedFilterRows = (CO / G) ;

int32_t reshapedFilterCols = ((FH * FW) * CIG) ;

int32_t reshapedIPRows = ((FH * FW) * CIG) ;

int32_t outH = ((((H + (zPadHLeft + zPadHRight)) - FH) / strideH) + 1) ;

int32_t outW = ((((W + (zPadWLeft + zPadWRight)) - FW) / strideW) + 1) ;

int32_t reshapedIPCols = ((N * outH) * outW) ;

for (uint32_t g = 0; g < G; g++){
auto inputReshaped = make_vector_float(ALICE, reshapedIPRows, reshapedIPCols) ;

auto matmulOP = make_vector_float(ALICE, reshapedFilterRows, reshapedIPCols) ;

auto filterReshaped = make_vector_float(ALICE, reshapedFilterRows, reshapedFilterCols) ;

Conv2DReshapeFilterGroup(FH, FW, CI, CO, g, G, filterArr, filterReshaped);
Conv2DReshapeInputGroup(N, H, W, CI, FH, FW, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, g, G, reshapedIPRows, reshapedIPCols, inputArr, inputReshaped);
MatMul2D(reshapedFilterRows, reshapedFilterCols, reshapedIPCols, filterReshaped, inputReshaped, matmulOP, 1);
Conv2DReshapeMatMulOPGroup(N, outH, outW, CO, g, G, matmulOP, outArr);
ClearMemSecret2(reshapedIPRows, reshapedIPCols, inputReshaped);
ClearMemSecret2(reshapedFilterRows, reshapedIPCols, matmulOP);
ClearMemSecret2(reshapedFilterRows, reshapedFilterCols, filterReshaped);
}
}

void Conv2DGroupWrapper(int32_t N, int32_t H, int32_t W, int32_t CI, int32_t FH, int32_t FW, int32_t CO, int32_t zPadHLeft, int32_t zPadHRight, int32_t zPadWLeft, int32_t zPadWRight, int32_t strideH, int32_t strideW, int32_t G, auto& inputArr, auto& filterArr, auto& outArr){
Conv2DGroup(N, H, W, CI, FH, FW, CO, zPadHLeft, zPadHRight, zPadWLeft, zPadWRight, strideH, strideW, G, inputArr, filterArr, outArr);
}


/*void GemmAdd(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& op, auto& bias){
for (uint32_t i1 = 0; i1 < s4; i1++){
for (uint32_t i2 = 0; i2 < s1; i2++){
for (uint32_t i3 = 0; i3 < s2; i3++){
for (uint32_t i4 = 0; i4 < s3; i4++){
op[i2][i3][i4][i1] = __fp_op->add(op[i2][i3][i4][i1], bias[i1]) ;
cout<<i1<<endl;

}
}
}
}
}*/

void GemmAdd(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& inArr, auto& bias) {
	int m_bits = inArr[0][0][0][0].m_bits ;
	int e_bits = inArr[0][0][0][0].e_bits ;
	int k ; 

	uint8_t *in_s = new uint8_t[s1*s2*s3*s4] ;
	uint8_t *in_z = new uint8_t[s1*s2*s3*s4] ;
	uint64_t *in_m = new uint64_t[s1*s2*s3*s4] ;
	uint64_t *in_e = new uint64_t[s1*s2*s3*s4] ;
	k = 0 ;
	for (int i = 0 ; i < s1 ; i++) {
	for (int j = 0 ; j < s2 ; j++) {
	for (int l = 0 ; l < s3 ; l++) {
		for (int m = 0 ; m < s4 ; m++) {
			in_s[k] = inArr[i][j][l][m].s[0] ;
			in_z[k] = inArr[i][j][l][m].z[0] ;
			in_m[k] = inArr[i][j][l][m].m[0] ;
			in_e[k] = inArr[i][j][l][m].e[0] ;
			k++ ;
			}
			}
		}
	}
	FPArray fp1 = __fp_op->input(__party, s1*s2*s3*s4, in_s, in_z, in_m, in_e, m_bits, e_bits) ;

	uint8_t *bias_s = new uint8_t[s1*s2*s3*s4] ;
	uint8_t *bias_z = new uint8_t[s1*s2*s3*s4] ;
	uint64_t *bias_m = new uint64_t[s1*s2*s3*s4] ;
	uint64_t *bias_e = new uint64_t[s1*s2*s3*s4] ;
	k = 0 ;
	for (int i = 0 ; i < s1*s2*s3 ; i++) {
		for (int j = 0 ; j < s4 ; j++) {
			bias_s[k] = bias[j].s[0] ;
			bias_z[k] = bias[j].z[0] ;
			bias_m[k] = bias[j].m[0] ;
			bias_e[k] = bias[j].e[0] ;
			k++ ;
		}
	}
	
	FPArray fp2 = __fp_op->input(__party, s1*s2*s3*s4, bias_s, bias_z, bias_m, bias_e, m_bits, e_bits) ;

	FPArray res = __fp_op->add(fp1, fp2) ;
	
	k = 0 ;
	for (int i = 0 ; i < s1 ; i++) {
	for (int j = 0 ; j < s2 ; j++) {
	for (int l = 0 ; l < s3 ; l++) {
		for (int m = 0 ; m < s4 ; m++) {  
			inArr[i][j][l][m].s[0] = res.s[k] ;
			inArr[i][j][l][m].z[0] = res.z[k] ;
			inArr[i][j][l][m].m[0] = res.m[k] ;
			inArr[i][j][l][m].e[0] = res.e[k] ;
			k++ ;
		}
	}
	}
	}

	delete[] in_s ; delete[] bias_s ; 
	delete[] in_z ; delete[] bias_z ; 
	delete[] in_m ; delete[] bias_m ; 
	delete[] in_e ; delete[] bias_e ;

}

void Hadamard1(int32_t s1, auto &inArr, auto &inArr1, auto &outArr) {
	int m_bits, e_bits ;
	m_bits = inArr[0].m_bits ;
	e_bits = inArr[0].e_bits ;

	uint8_t *in_s = new uint8_t[s1] ;
	uint8_t *in_z = new uint8_t[s1] ;
	uint64_t *in_m = new uint64_t[s1] ;
	uint64_t *in_e = new uint64_t[s1] ;

	for (int i = 0 ; i < s1 ; i++) {
		in_s[i] = inArr[i].s[0] ;
		in_z[i] = inArr[i].z[0] ;
		in_m[i] = inArr[i].m[0] ;
		in_e[i] = inArr[i].e[0] ;
	}

	FPArray in_flat = __fp_op->input(__party, s1, in_s, in_z, in_m, in_e, m_bits, e_bits) ;
	
	uint8_t *in_s1 = new uint8_t[s1] ;
	uint8_t *in_z1 = new uint8_t[s1] ;
	uint64_t *in_m1 = new uint64_t[s1] ;
	uint64_t *in_e1 = new uint64_t[s1] ;

	for (int i = 0 ; i < s1 ; i++) {
		in_s1[i] = inArr1[i].s[0] ;
		in_z1[i] = inArr1[i].z[0] ;
		in_m1[i] = inArr1[i].m[0] ;
		in_e1[i] = inArr1[i].e[0] ;
	}
	
	FPArray zero_flat = __fp_op->input(__party, s1, in_s1, in_z1, in_m1, in_e1, m_bits, e_bits) ;

	FPArray out_flat = __fp_op->mul(in_flat, zero_flat) ;

	for (int i = 0 ; i < s1 ; i++) {
		outArr[i].s[0] = out_flat.s[i] ;
		outArr[i].z[0] = out_flat.z[i] ;
		outArr[i].m[0] = out_flat.m[i] ;
		outArr[i].e[0] = out_flat.e[i] ;
		// outArr[i] = out_flat.subset(i, i+1) ;
	}

	delete[] in_s ; 
	delete[] in_z ; 
	delete[] in_m ;  
	delete[] in_e ; 
	delete[] in_s1 ; 
	delete[] in_z1 ; 
	delete[] in_m1 ;  
	delete[] in_e1 ; 
}


void Hadamard(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& inArr, auto& inArr1, auto& outArr){

	int32_t size = (s1 * s2* s3* s4) ;
	auto reshapedInArr = make_vector_float(ALICE, size) ;
	auto reshapedInArr1 = make_vector_float(ALICE, size) ;
	auto reshapedOutArr = make_vector_float(ALICE, size) ;

	for (uint32_t i1 = 0, linIdx=0; i1 < s1; i1++){
	for (uint32_t i2 = 0; i2 < s2; i2++){
	for (uint32_t i3 = 0; i3 < s3; i3++){
	for (uint32_t i4 = 0; i4 < s4; i4++, linIdx++){

	reshapedInArr[linIdx] = inArr[i1][i2][i3][i4] ;
	
	}
	}
	}
	}
	
	for (uint32_t i1 = 0, linIdx=0; i1 < s1; i1++){
	for (uint32_t i2 = 0; i2 < s2; i2++){
	for (uint32_t i3 = 0; i3 < s3; i3++){
	for (uint32_t i4 = 0; i4 < s4; i4++, linIdx++){

	reshapedInArr1[linIdx] = inArr1[i1][i2][i3][i4] ;
	
	}
	}
	}
	}
	
	Hadamard1(size, reshapedInArr, reshapedInArr1, reshapedOutArr);
	
	for (uint32_t i1 = 0, linIdx=0; i1 < s1; i1++){
	for (uint32_t i2 = 0; i2 < s2; i2++){
	for (uint32_t i3 = 0; i3 < s3; i3++){
	for (uint32_t i4 = 0; i4 < s4; i4++, linIdx++){
	
	outArr[i1][i2][i3][i4] = reshapedOutArr[linIdx] ;
	}
	}
	}
	}
	ClearMemSecret1(size, reshapedInArr);
	ClearMemSecret1(size, reshapedOutArr);
}




void Add(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& arr1, auto& arr2, auto& outArr){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
outArr[i1][i2][i3][i4] = __fp_op->add(arr1[i1][i2][i3][i4], arr2[i1][i2][i3][i4]) ;

}
}
}
}
}


void Sigmoid1(int32_t s1, auto &inArr, auto &outArr) {
	int m_bits, e_bits ;
	m_bits = inArr[0].m_bits ;
	e_bits = inArr[0].e_bits ;

	uint8_t *in_s = new uint8_t[s1] ;
	uint8_t *in_z = new uint8_t[s1] ;
	uint64_t *in_m = new uint64_t[s1] ;
	uint64_t *in_e = new uint64_t[s1] ;

	for (int i = 0 ; i < s1 ; i++) {
		in_s[i] = inArr[i].s[0] ;
		in_z[i] = inArr[i].z[0] ;
		in_m[i] = inArr[i].m[0] ;
		in_e[i] = inArr[i].e[0] ;
	}

	FPArray in_flat = __fp_op->input(__party, s1, in_s, in_z, in_m, in_e, m_bits, e_bits) ;

	float zero = 0.0 ;
	FPArray zero_flat = __fp_op->input<float>(ALICE, s1, zero, m_bits, e_bits) ;
	float one = 1.0 ;
	FPArray one_flat = __fp_op->input<float>(ALICE, s1, one, m_bits, e_bits) ;

	FPArray out_flat = __fp_op->sub(zero_flat, in_flat) ;
	out_flat = __fp_math->exp(out_flat) ;
	out_flat = __fp_op->add(one_flat, out_flat) ;
	out_flat = __fp_op->div(one_flat, out_flat) ;

	for (int i = 0 ; i < s1 ; i++) {
		outArr[i].s[0] = out_flat.s[i] ;
		outArr[i].z[0] = out_flat.z[i] ;
		outArr[i].m[0] = out_flat.m[i] ;
		outArr[i].e[0] = out_flat.e[i] ;
		// outArr[i] = out_flat.subset(i, i+1) ;
	}

	delete[] in_s ; 
	delete[] in_z ; 
	delete[] in_m ;  
	delete[] in_e ; 
}

void Sigmoid4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& inArr, auto& outArr){

	int32_t size = (s1 * s2* s3* s4) ;
	auto reshapedInArr = make_vector_float(ALICE, size) ;
	auto reshapedOutArr = make_vector_float(ALICE, size) ;

	for (uint32_t i1 = 0, linIdx=0; i1 < s1; i1++){
	for (uint32_t i2 = 0; i2 < s2; i2++){
	for (uint32_t i3 = 0; i3 < s3; i3++){
	for (uint32_t i4 = 0; i4 < s4; i4++, linIdx++){

	reshapedInArr[linIdx] = inArr[i1][i2][i3][i4] ;
	
	}
	}
	}
	}
	
	Sigmoid1(size, reshapedInArr, reshapedOutArr);
	
	for (uint32_t i1 = 0, linIdx=0; i1 < s1; i1++){
	for (uint32_t i2 = 0; i2 < s2; i2++){
	for (uint32_t i3 = 0; i3 < s3; i3++){
	for (uint32_t i4 = 0; i4 < s4; i4++, linIdx++){
	
	outArr[i1][i2][i3][i4] = reshapedOutArr[linIdx] ;
	}
	}
	}
	}
	ClearMemSecret1(size, reshapedInArr);
	ClearMemSecret1(size, reshapedOutArr);

}

void Tanh1(int32_t s1, auto &inArr, auto &outArr) {
	int m_bits, e_bits ;
	m_bits = inArr[0].m_bits ;
	e_bits = inArr[0].e_bits ;

	uint8_t *in_s = new uint8_t[s1] ;
	uint8_t *in_z = new uint8_t[s1] ;
	uint64_t *in_m = new uint64_t[s1] ;
	uint64_t *in_e = new uint64_t[s1] ;

	for (int i = 0 ; i < s1 ; i++) {
		in_s[i] = inArr[i].s[0] ;
		in_z[i] = inArr[i].z[0] ;
		in_m[i] = inArr[i].m[0] ;
		in_e[i] = inArr[i].e[0] ;
	}

	FPArray num = __fp_op->input(__party, s1, in_s, in_z, in_m, in_e, m_bits, e_bits) ;

	float zero = 0.0 ;
	FPArray zero_flat = __fp_op->input<float>(ALICE, s1, zero, m_bits, e_bits) ;

	FPArray neg_num = __fp_op->sub(zero_flat, num) ;
	neg_num = __fp_math->exp(neg_num) ;
	num = __fp_math->exp(num) ;
	FPArray out_flat =__fp_op->div(__fp_op->sub(num, neg_num), __fp_op->add(num, neg_num)); 

	for (int i = 0 ; i < s1 ; i++) {
		outArr[i].s[0] = out_flat.s[i] ;
		outArr[i].z[0] = out_flat.z[i] ;
		outArr[i].m[0] = out_flat.m[i] ;
		outArr[i].e[0] = out_flat.e[i] ;
		// outArr[i] = out_flat.subset(i, i+1) ;
	}

	delete[] in_s ; 
	delete[] in_z ; 
	delete[] in_m ;  
	delete[] in_e ; 
}


void Tanh4(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& inArr, auto& outArr){

	int32_t size = (s1 * s2* s3* s4) ;
	auto reshapedInArr = make_vector_float(ALICE, size) ;
	auto reshapedOutArr = make_vector_float(ALICE, size) ;

	for (uint32_t i1 = 0, linIdx=0; i1 < s1; i1++){
	for (uint32_t i2 = 0; i2 < s2; i2++){
	for (uint32_t i3 = 0; i3 < s3; i3++){
	for (uint32_t i4 = 0; i4 < s4; i4++, linIdx++){

	reshapedInArr[linIdx] = inArr[i1][i2][i3][i4] ;
	
	}
	}
	}
	}
	
	Tanh1(size, reshapedInArr, reshapedOutArr);
	
	for (uint32_t i1 = 0, linIdx=0; i1 < s1; i1++){
	for (uint32_t i2 = 0; i2 < s2; i2++){
	for (uint32_t i3 = 0; i3 < s3; i3++){
	for (uint32_t i4 = 0; i4 < s4; i4++, linIdx++){
	
	outArr[i1][i2][i3][i4] = reshapedOutArr[linIdx] ;
	}
	}
	}
	}
	ClearMemSecret1(size, reshapedInArr);
	ClearMemSecret1(size, reshapedOutArr);

}

void InitHiddenStates(int32_t s1, int32_t s2, int32_t s3, int32_t s4, auto& Ht){
for (uint32_t i1 = 0; i1 < s1; i1++){
for (uint32_t i2 = 0; i2 < s2; i2++){
for (uint32_t i3 = 0; i3 < s3; i3++){
for (uint32_t i4 = 0; i4 < s4; i4++){
Ht[i1][i2][i3][i4] = __public_float_to_arithmetic(0.) ;

}
}
}
}
}

void forward(auto& layer1W, auto& XH, auto& b, auto& Ct1, auto& Ht){

auto outputarr = make_vector_float(ALICE, 1, 5, 5, 20) ;
auto outputsigmoid = make_vector_float(ALICE, 1, 5, 5, 20) ;
auto igate = make_vector_float(ALICE, 1, 5, 5, 5) ;
auto fgate = make_vector_float(ALICE, 1, 5, 5, 5) ;
auto ogate = make_vector_float(ALICE, 1, 5, 5, 5) ;
auto ggate = make_vector_float(ALICE, 1, 5, 5, 5) ;
auto outputg = make_vector_float(ALICE, 1, 5, 5, 5) ;
auto add1 = make_vector_float(ALICE, 1, 5, 5, 5) ;
auto add2 = make_vector_float(ALICE, 1, 5, 5, 5) ;
auto cnextupdated = make_vector_float(ALICE, 1, 5, 5, 5) ;

Conv2DGroupWrapper(1, 5, 5, 6, 3, 3, 20, 1, 1, 1, 1, 1, 1, 1, XH, layer1W, outputarr);
GemmAdd(1, 5, 5, 20, outputarr, b);

/*for (uint32_t i0 = 0; i0 < 1; i0++){
for (uint32_t i3 = 0; i3 < 20; i3++){
for (uint32_t i1 = 0; i1 < 5; i1++){
for (uint32_t i2 = 0; i2 < 5; i2++){
	FPArray print_=__fp_op->output(PUBLIC, outputarr[i0][i1][i2][i3]);
	vector<float> f_=print_.get_native_type<float>();
	cout<<f_[0]<<" "; } } } } */

Sigmoid4(1, 5, 5, 15, outputarr, outputsigmoid);
for (uint32_t l = 0; l < 1; l++)
for (uint32_t i = 0; i < 5; i++)
for (uint32_t j = 0; j < 5; j++)
for (uint32_t k = 0; k < 5; k++){
igate[l][i][j][k] = outputsigmoid[l][j][k][i] ;
fgate[l][i][j][k] = outputsigmoid[l][j][k][(5 + i)] ;
ogate[l][i][j][k] = outputsigmoid[l][j][k][(10 + i)] ;
ggate[l][i][j][k] = outputarr[l][j][k][(15 + i)] ;
}

Tanh4(1, 5, 5, 5, ggate, outputg);
Hadamard(1, 5, 5, 5, fgate, Ct1, add1);
Hadamard(1, 5, 5, 5, igate, outputg, add2);
Add(1, 5, 5, 5, add1, add2, Ct1);
Tanh4(1, 5, 5, 5, Ct1, cnextupdated);
Hadamard(1, 5, 5, 5, ogate, cnextupdated, Ht);

}



int main (int __argc, char **__argv) {
__init(__argc, __argv) ;

auto inp = make_vector_float(ALICE, 1, 5, 5, 6) ;

if ((__party == BOB)) {
cout << ("Input inp:") << endl ;

}
float *__tmp_in_inp = new float[1] ;

for (uint32_t i0 = 0; i0 < 1; i0++){
for (uint32_t i3 = 0; i3 < 6; i3++){
for (uint32_t i1 = 0; i1 < 5; i1++){
for (uint32_t i2 = 0; i2 < 5; i2++){
if ((__party == BOB)) {
cin >> __tmp_in_inp[0];
}
inp[i0][i1][i2][i3] = __fp_op->input(BOB, 1, __tmp_in_inp, __m_bits, __e_bits) ;

}
}
}
}
delete[] __tmp_in_inp ;

auto Ct1 = make_vector_float(ALICE, 1, 5, 5, 5) ;

if ((__party == BOB)) {
cout << ("Input Ct1:") << endl ;

}
float *__tmp_in_Ct1 = new float[1] ;

for (uint32_t i0 = 0; i0 < 1; i0++){
for (uint32_t i1 = 0; i1 < 5; i1++){
for (uint32_t i2 = 0; i2 < 5; i2++){
for (uint32_t i3 = 0; i3 < 5; i3++){
if ((__party == BOB)) {
cin >> __tmp_in_Ct1[0];
}
Ct1[i0][i1][i2][i3] = __fp_op->input(BOB, 1, __tmp_in_Ct1, __m_bits, __e_bits) ;
}
}
}
}

if ((__party == BOB)) {
cout << ("Input Ht:") << endl ;

}

auto Ht = make_vector_float(ALICE, 1, 5, 5, 5) ;

for (uint32_t i0 = 0; i0 < 1; i0++){
for (uint32_t i1 = 0; i1 < 5; i1++){
for (uint32_t i2 = 0; i2 < 5; i2++){
for (uint32_t i3 = 0; i3 < 5; i3++){
if ((__party == BOB)) {
cin >> __tmp_in_Ct1[0];
}
Ht[i0][i1][i2][i3] = __fp_op->input(BOB, 1, __tmp_in_Ct1, __m_bits, __e_bits) ;
}
}
}
}
delete[] __tmp_in_Ct1 ;

auto W = make_vector_float(ALICE, 3, 3, 6, 20) ;

if ((__party == ALICE)) {
cout << ("Input W:") << endl ;

}
float *__tmp_in_W = new float[1] ;


for (uint32_t i3 = 0; i3 < 20; i3++){
for (uint32_t i2 = 0; i2 < 6; i2++){
for (uint32_t i0 = 0; i0 < 3; i0++){
for (uint32_t i1 = 0; i1 < 3; i1++){
if ((__party == ALICE)) {
cin >> __tmp_in_W[0];
}
W[i0][i1][i2][i3] = __fp_op->input(ALICE, 1, __tmp_in_W, __m_bits, __e_bits) ;
} } } }
delete[] __tmp_in_W ;

auto bias = make_vector_float(ALICE, 20) ;

if ((__party == ALICE)) {
cout << ("Input bias:") << endl ;

}
float *__tmp_in_bias = new float[1] ;

for (uint32_t i0 = 0; i0 < 20; i0++){
if ((__party == ALICE)) {
cin >> __tmp_in_bias[0];
}
bias[i0] = __fp_op->input(ALICE, 1, __tmp_in_bias, __m_bits, __e_bits) ;

}
delete[] __tmp_in_bias ;

int32_t iters = 1 ;

//InitHiddenStates(1, 5, 5, 5, Ht);
auto start = clock_start();
for (uint32_t i = 0; i < iters; i++){
forward(W, inp, bias, Ct1, Ht);
}
long long t = time_from(start);
cout << "Total Time for Full Run:\t" << t / (1000.0) << " ms" << endl;
return 0;
}
