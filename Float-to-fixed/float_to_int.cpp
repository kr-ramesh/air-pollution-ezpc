#include "FloatingPoint/floating-point.h"
#include "FloatingPoint/fp-math.h"
#include <random>
#include <limits>
#include <iostream>
// #include "float_utils.h"
#include <omp.h>

using namespace sci;
using namespace std;


IOPack *iopack = nullptr;
OTPack *otpack = nullptr;
FPOp *fp_op = nullptr;
FPMath *fp_math = nullptr;
FixOp *fix = nullptr;
BoolOp *bool_op= nullptr;
int party = 1;
int bit_len=32;
string address = "127.0.0.1";
int port = 8000;
uint8_t m_bits = 23, e_bits =8;

tuple<BoolArray,BoolArray,FixArray,FixArray> get_components(const FPArray &x) {
  BoolArray x_s = bool_op->input(x.party, x.size, x.s);
  BoolArray x_z = bool_op->input(x.party, x.size, x.z);
  FixArray x_m = fix->input(x.party, x.size, x.m, false, x.m_bits + 1, x.m_bits);
  FixArray x_e = fix->input(x.party, x.size, x.e, true, x.e_bits + 2, 0);
  return make_tuple(x_s, x_z, x_m, x_e);
}

int64_t get_int(FixArray &other, uint64_t i) {
  return (other.signed_ ? signed_val(other.data[i], other.ell) : other.data[i]);
}

float float_gen(float Min, float Max)
{
    return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
}

uint64_t do_shift(int s) {
  return (((uint64_t)1) << s) ;
}

int main(int argc, char **argv) {
  cout.precision(15);

  ArgMapping amap;

  amap.arg("r", party, "Role of party: ALICE/SERVER = 1; BOB/CLIENT = 2");
  amap.arg("p", port, "Port Number");
  amap.arg("ip", address, "IP Address of server (ALICE)");
  amap.parse(argc, argv);
  
  iopack = new IOPack(party, port, address);
  otpack = new OTPack(iopack, party);
    
  srand(time(0));  // Initialize random number generator.

  fp_op = new FPOp(party, iopack, otpack);
  fp_math = new FPMath(party, iopack, otpack);
  fix = new FixOp(party, iopack, otpack);
  bool_op= new BoolOp(party, iopack, otpack);
  auto start = clock_start();
  uint64_t comm_start = iopack->get_comm();
  uint64_t initial_rounds = iopack->get_rounds();
    
  //vector for the generation of the LUT
  vector<uint64_t> spec_vec_v0;
  for(int i=0;i<64;i++)
  	spec_vec_v0.insert(spec_vec_v0.end(), pow(2, i));
  	
  //Sample numbers for testing
  // float f_1[]={4.9042821504535347,9.00000001,0,22.0507667243214,-99.6175791042,22.244144885, 88.3889394614,67.5236281643,74.8189688697,4.840235094,11.0652513346,69.1172942636,65.3377424597,74.3603258759,68.8458678959,22.0030407771,39.144387384,93.5312618825,23.5384549711,52.0022613425};
  vector<float> f_1 ;
  uint32_t num = do_shift(24) ;
  float f ;
  for (uint64_t i = 0 ; i <= do_shift(8) ; i += 1) {
    num++ ;
    f = *((float*)&num) ;

    if (isnormal(f))
      f_1.push_back(f) ;
  }

    
  //Initialization of array with test inputs, the FPArray containing these test inputs (size is set to 20)
  uint64_t sz = (uint64_t)f_1.size() ;
  FPArray x = fp_op->input<float>(ALICE, sz, &f_1[0], m_bits, e_bits);
  BoolArray x_s, x_z, is_sign, is_greater_than_23, is_greater_than_0;
  FixArray x_m, x_e, check, print_;
  
  //SZME format is extracted
  tie(x_s, x_z, x_m, x_e) = get_components(x);
  
  //Returns true if the sign bit has been set to 1, else 0
  is_sign=bool_op->AND(bool_op->input(ALICE, sz, 1), x_s);
  
  //check stores the value of the exp+scale
  check=fix->sub(x_e, int(127));
  
  //Evaluates for which of the numbers (exp-23) is greater than 0
  is_greater_than_23=fix->GT(check, 23);
  
  //Evaluates for which of the numbers exp is greater than -1 (If true, the number approximates to 0) 
  is_greater_than_0=fix->GT(check, -1);
  
  //initialization of FixArray with mantissa components and idx for the values corresponding to exp+scale in the lookup table
  FixArray a=fix->input(ALICE, sz, x_m.data, true, bit_len, 0);
  
  // LUT generated for 2^n
  FixArray idx=fix->input(ALICE, sz, check.data, false, bit_len, 0);
  FixArray v0 = fix->LUT(spec_vec_v0, idx, false, 63, 0, 6);
  
  //Initializes v0 to 0 so that the output is 0 when is_greater_than_0 is not satisfied
  v0= fix->if_else(is_greater_than_0, v0, 0);
  
  //Represents the value of m*2^exp. v0 obtained from the lookup table. Condition: if exp-23>0, right shift v0 by 23 before multiplying it with 'a'.
  FixArray result = fix->if_else(is_greater_than_23, fix->mul(a, fix->right_shift(v0, int(23)), 63),fix->right_shift(fix->mul(a, v0, 63), int(23)));
  
  //Result is negative if the sign bit has been set, else positive
  result=fix->if_else(is_sign, fix->mul(result, uint64_t(-1), 63), result);

  
  //Prints the resulting FixArray. Checks if the result concurs with the values stored in real_value (this is for testing purposes only)  
  print_=fix->output(PUBLIC, result);

  uint64_t comm_end = iopack->get_comm();
  long long t = time_from(start);

  bool chk = true ;
  // cout << "(Secure, Clear) - \n" ;
  for (int i = 0 ; i < sz ; i++) {
    // printf("(%lld, %lld)\n", get_int(print_, i), int64_t(f_1[i])) ;
    chk = chk && (get_int(print_, i) == int64_t(f_1[i])) ;
  }

  cout << "sz = " << sz << endl ;
  cout << "CHK = " << chk << endl ;
  cout << "Comms = " << (comm_end - comm_start) << " bytes " << endl ;
  // cout << "Number of FP ops/s:\t" << (double(sz) / t) * 1e6 << std::endl;
  cout << "Total Time = " << t / (1000.0) << " ms" << endl;
  cout << "Num_rounds = " << (iopack->get_rounds() - initial_rounds) << endl;

  delete iopack;
  delete otpack;
  delete fp_op;
  delete fp_math;
}


