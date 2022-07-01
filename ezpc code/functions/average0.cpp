#include "emp-sh2pc/emp-sh2pc.h" 

//#include <algorithm>
#include "../../install/include/abycore/sharing/arithsharing.h"

//#include "../aby/abysetup.h"

using namespace emp;
using namespace std;
int bitlen = 16;
int party,port;
char *ip = "127.0.0.1"; 
template<typename T> 
vector<T> make_vector(size_t size) { 
return std::vector<T>(size); 
} 

template <typename T, typename... Args> 
auto make_vector(size_t first, Args... sizes) 
{ 
auto inner = make_vector<T>(sizes...); 
return vector<decltype(inner)>(first, inner); 
} 

const uint32_t dim =  (int32_t)270;

Bit check1(auto& x2, Integer threshold, Bit result){
Integer count = Integer(bitlen,  (int32_t)0, PUBLIC);
for (uint32_t i =  (int32_t)0; i <  (int32_t)270; i++){
/* Temporary variable for sub-expression on source location: (8,22-8,27) */
Bit __tac_var1 = x2[i];
/* Temporary variable for sub-expression on source location: (8,31-8,32) */
Integer __tac_var2 = Integer(bitlen,  (int32_t)1, PUBLIC);
/* Temporary variable for sub-expression on source location: (8,35-8,36) */
Integer __tac_var3 = Integer(bitlen,  (int32_t)0, PUBLIC);
/* Temporary variable for sub-expression on source location: (8,21-8,36) */
Integer __tac_var4 =  If(__tac_var1, __tac_var2, __tac_var3);
count = count.operator+(__tac_var4);
}
result = count.operator>(threshold);
return result;
}

Integer sum(auto& x1, auto& x2, Integer resultfinal, Bit ischeck){
for (uint32_t i =  (int32_t)0; i <  (int32_t)270; i++){
/* Temporary variable for sub-expression on source location: (17,46-17,51) */
Bit __tac_var5 = x2[i];
/* Temporary variable for sub-expression on source location: (17,55-17,60) */
Integer __tac_var6 = x1[i];
/* Temporary variable for sub-expression on source location: (17,63-17,64) */
Integer __tac_var7 = x1[i]-x1[i];
/* Temporary variable for sub-expression on source location: (17,45-17,64) */
Integer __tac_var8 =  If(__tac_var5, __tac_var6, __tac_var7);
/* Temporary variable for sub-expression on source location: (17,30-17,65) */
/* Temporary variable for sub-expression on source location: (17,69-17,70) */
resultfinal =  resultfinal+__tac_var8;
}
return resultfinal;
}


int main(int argc, char** argv) {
parse_party_and_port(argv, &party, &port);
if(argc>3){
  ip=argv[3];
}
cout<<"Ip Address: "<<ip<<endl;
cout<<"Port: "<<port<<endl;
cout<<"Party: "<<(party==1? "CLIENT" : "SERVER")<<endl;
NetIO * io = new NetIO(party==ALICE ? nullptr : ip, port);
setup_semi_honest(io, party);


auto grid1 = make_vector<Integer>( (int32_t)270);
if ((party == BOB)) {
cout << ("Input grid1:") << endl;
}
/* Variable to read the clear value corresponding to the input variable grid1 at (23,2-23,37) */
uint32_t __tmp_in_grid1;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)270; i0++){
if ((party == BOB)) {
cin >> __tmp_in_grid1;
}
grid1[i0] = Integer(bitlen, __tmp_in_grid1, BOB);
}

auto bool1 = make_vector<Bit>( (int32_t)270);
if ((party == BOB)) {
cout << ("Input bool1:") << endl;
}
/* Variable to read the clear value corresponding to the input variable bool1 at (24,2-24,36) */
bool __tmp_in_bool1;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)270; i0++){
if ((party == BOB)) {
cin >> __tmp_in_bool1;
}
bool1[i0] = Bit(__tmp_in_bool1, BOB);
}

Integer randnum;
if ((party == BOB)) {
cout << ("Input randnum:") << endl;
}
/* Variable to read the clear value corresponding to the input variable randnum at (25,2-25,34) */
uint32_t __tmp_in_randnum;
if ((party == BOB)) {
cin >> __tmp_in_randnum;
}
randnum = Integer(bitlen, __tmp_in_randnum, BOB);

auto grid2 = make_vector<Integer>( (int32_t)270);
if ((party == ALICE)) {
cout << ("Input grid2:") << endl;
}
/* Variable to read the clear value corresponding to the input variable grid2 at (27,2-27,37) */
uint32_t __tmp_in_grid2;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)270; i0++){
if ((party == ALICE)) {
cin >> __tmp_in_grid2;
}
grid2[i0] = Integer(bitlen, __tmp_in_grid2, ALICE);
}

auto bool2 = make_vector<Bit>( (int32_t)270);
if ((party == ALICE)) {
cout << ("Input bool2:") << endl;
}
/* Variable to read the clear value corresponding to the input variable bool2 at (28,2-28,36) */
bool __tmp_in_bool2;
for (uint32_t i0 =  (uint32_t)0; i0 <  (int32_t)270; i0++){
if ((party == ALICE)) {
cin >> __tmp_in_bool2;
}
bool2[i0] = Bit(__tmp_in_bool2, ALICE);
}

auto grid = make_vector<Integer>( (int32_t)270);
for (uint32_t i =  (int32_t)0; i <  (int32_t)270; i++){
/* Temporary variable for sub-expression on source location: (39,12-39,20) */
Integer __tac_var11 = grid1[i];
/* Temporary variable for sub-expression on source location: (39,21-39,29) */
Integer __tac_var12 = grid2[i];
grid[i] = __tac_var11.operator+(__tac_var12);
}

auto clientbool = make_vector<Bit>( (int32_t)270);

Integer thres = Integer(bitlen,  (int32_t)1, PUBLIC);
for (uint32_t i =  (int32_t)0; i <  (int32_t)270; i++){
/* Temporary variable for sub-expression on source location: (48,18-48,26) */
Bit __tac_var13 = bool1[i];
/* Temporary variable for sub-expression on source location: (48,27-48,35) */
Bit __tac_var14 = bool2[i];
clientbool[i] = __tac_var13.operator^(__tac_var14);
}

Integer result=Integer(bitlen, (int32_t)0, PUBLIC);

Bit firstcheck;
auto start = clock_start();
firstcheck=check1(clientbool, thres, firstcheck);
result=If(firstcheck, sum(grid, clientbool, result, firstcheck), Integer(bitlen, (int32_t)0, PUBLIC));
/* Temporary variable for sub-expression on source location: (58,17-58,31) */
Integer __tac_var15 = result.operator-(randnum);
cout << ("Value of __tac_var15:") << endl;
cout << (__tac_var15.reveal<int32_t>(ALICE)) << endl;
long long t = time_from(start);
cout << "Total Time:\t" << t / (1000.0) << " ms" << endl;


finalize_semi_honest();
delete io; 
 
return 0;
}