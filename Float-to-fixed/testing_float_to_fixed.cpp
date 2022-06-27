#include "FloatingPoint/floating-point.h"
#include "FloatingPoint/fp-math.h"
#include <random>
#include <limits>
#include "float_utils.h"
#include <omp.h>

using namespace sci;
using namespace std;


IOPack *iopack = nullptr;
OTPack *otpack = nullptr;
FPOp *fp_op = nullptr;
FPMath *fp_math = nullptr;
FixOp *fix = nullptr;
BoolOp *bool_op= nullptr;
int sz = 20;
int party = 1;
int bit_len=63;
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

float float_gen(float Min, float Max)
{
    return ((float(rand()) / float(RAND_MAX)) * (Max - Min)) + Min;
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
  
  //NetIO *io = new NetIO(party == 1 ? nullptr : address.c_str(), port);
  
  /*
  if(party==2)
  cin>>scale_factor[0];

  if(party==2)
  io->send_data(scale_factor, sizeof(float));
  if(party==1)
  io->recv_data(scale_factor, sizeof(float));*/
  srand(time(0));  // Initialize random number generator.

  float scale_factor[1];
  fp_op = new FPOp(party, iopack, otpack);
  fp_math = new FPMath(party, iopack, otpack);
  fix = new FixOp(party, iopack, otpack);
  bool_op= new BoolOp(party, iopack, otpack);
  auto start = clock_start();
  uint64_t comm_start = iopack->get_comm();
  uint64_t initial_rounds = iopack->get_rounds();
  
  vector<uint64_t> spec_vec_v0;
  for(int i=0;i<64;i++)
  	spec_vec_v0.insert(spec_vec_v0.end(), pow(2, i));

  //float f_1[]={4.9042821504535347,-70.40441973423401,0,77.0507667243214,-99.6175791042,22.244144885,88.3889394614,67.5236281643,74.8189688697,4.840235094,11.0652513346,69.1172942636,65.3377424597,74.3603258759,68.8458678959,22.0030407771,39.144387384,93.5312618825,23.5384549711,52.0022613425};
  uint64_t iterations=0;
  float* f_1 = new float[sz];
  float* real_val = new float[sz];
  while(iterations<=1000)
  {
  //generates randon numbers for the scale within this range (this is for testing purposes only)
  scale_factor[0]=(rand() % 23) + -1;
  
  //generates random numbers to be converted to fixed point (this is for testing purposes only)
  for(int i=0;i<sz;i++)
  f_1[i]=float_gen(-1000,1000);

  //to generate the value to check if the conversion is correct (this is for testing purposes only)
  for(int i=0;i<sz;i++)
  real_val[i]=int(pow(2, scale_factor[0])*f_1[i]);
  
  //Initialization of array with test inputs, the FPArray containing these test inputs (size is set to 20)
  FPArray x = fp_op->input<float>(ALICE, sz, f_1, m_bits, e_bits);
  BoolArray x_s, x_z, is_sign;
  FixArray x_m, x_e, check;
  //SZME format is extracted
  tie(x_s, x_z, x_m, x_e) = get_components(x);
  
  //Returns true if the sign bit has been set to 1, else 0
  is_sign=bool_op->AND(bool_op->input(ALICE, sz, 1), x_s);
  
  //check stores the value of the exp+scale
  check=fix->sub(fix->add(x_e, scale_factor[0]), int(127));
  
  
  //initialization of FixArray with mantissa components and idx for the values corresponding to exp+scale in the lookup table
  FixArray a=fix->input(ALICE, sz, x_m.data, true, bit_len, 0);
  FixArray idx=fix->input(ALICE, sz, check.data, false, bit_len, 0);

  //LUT generated for 2^n
  FixArray v0 = fix->LUT(spec_vec_v0, idx, false, bit_len, 0, 6);

  //Represents the value of m*2^(exp+scale). v0 obtained from the lookup table
  FixArray result = fix->mul(a, v0, bit_len);
  
  //Right shift of the result of m*2^(exp+scale) by 23 bits
  result=fix->right_shift(result, int(23));
  
  //Result is negative if the sign bit has been set, else positive
  result=fix->if_else(is_sign, fix->mul(result, uint64_t(-1), bit_len), result);

  //Prints the resulting FixArray. Checks if the result concurs with the values stored in real_value (this is for testing purposes only)
  FixArray other= fix->output(PUBLIC, result);
  for (int i = 0; i < sz; i++){
  int64_t data_ =(other.signed_ ? signed_val(other.data[i], other.ell) : other.data[i]);
  if(data_!=real_val[i])
  cout<<"Error:"<<data_<<" "<<real_val[i]<<endl;
  }
  
  iterations+=1;
  }
  uint64_t comm_end = iopack->get_comm();
  long long t = time_from(start);

  //for(int i=0;i<sz;i++)
  //cout<<int(pow(2, scale_factor[0])*f_1[i])<<endl; 
  
  cout << "Comm. per operations: " << 8 * (comm_end - comm_start) / sz
       << " bits" << endl;
  cout << "Number of FP ops/s:\t" << (double(sz) / t) * 1e6 << std::endl;
  cout << "Total Time:\t" << t / (1000.0) << " ms" << endl;
  cout << "Num_rounds: " << (iopack->get_rounds() - initial_rounds) << endl;


  delete iopack;
  delete otpack;
  delete fp_op;
  delete fp_math;
}

