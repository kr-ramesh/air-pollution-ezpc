#include <iostream>
#include <random>
#include <tuple>
#include <bitset>

using namespace std ;

#define EBITS 8
#define MBITS 23

#define NUM_S 1
#define NUM_E (1 << 8)
#define NUM_M (1 << 23)

uint32_t s_mask = 0b10000000000000000000000000000000 ;
uint32_t e_mask = 0b01111111100000000000000000000000 ;
uint32_t m_mask = 0b00000000011111111111111111111111 ;

// Shift to appropriate position

uint32_t shift_to_s(uint32_t x) {
  return x << (EBITS + MBITS) ;
}

uint32_t shift_to_e(uint32_t x) {
  return x << MBITS ;
}

uint32_t shift_to_m(uint32_t x) {
  return x ;
}

// Shift from appropriate position to the LSBs

uint32_t shift_from_s(uint32_t x) {
  return x >> (EBITS + MBITS) ;
}

uint32_t shift_from_e(uint32_t x) {
  return x >> MBITS ;
}

uint32_t shift_from_m(uint32_t x) {
  return x ;
}

// Construction and deconstruction functions

float construct_float(uint32_t s, uint32_t e, uint32_t m) {
  uint32_t f = 0 ;

  f |= shift_to_s(s) ;
  f |= shift_to_e(e) ;
  f |= shift_to_m(m) ;

  return *((float*)&f) ;
}

tuple<uint32_t, uint32_t, uint32_t> deconstruct_float(float f) {
  uint32_t x = *((uint32_t*)&f) ;
  uint32_t s, e, m ;

  s = shift_from_s(x & s_mask) ;
  e = shift_from_e(x & e_mask) ;
  m = shift_from_m(x & m_mask) ;

  return make_tuple(s, e, m) ;
}

uint32_t mantissa_from_m(uint32_t m) {
  return m | (1 << MBITS) ;
}

// Float to int converter

int64_t float_to_int(uint32_t s, uint32_t e, uint32_t m) {
  // Set sign bit
  bool is_sign = s ;

  // Subtract bias to get unbiased exponent
  int32_t check = e - 127 ;

  //Checks if exp >= 23
  bool is_greater_than_23 = check > 23 ;

  //Evaluates for which of the numbers exp is greater than -1 (If false, the number approximates to 0) 
  bool is_greater_than_0 = check > -1 ;

  // Get mantissa
  uint64_t a = mantissa_from_m(m) ;

  // 2^e
  uint64_t v0 = (check < 0) ? 0 : (uint64_t(1) << check) ;

  // If exponent is too small, then result is 0.
  v0 = is_greater_than_0 ? v0 : 0 ;

  // Final result
  uint64_t res = is_greater_than_23 ? (a * (v0 >> 23)) : ((a * v0) >> 23) ;

  // Multiply with sign if required
  return (is_sign ? int64_t(-1) : int64_t(1)) * ((int64_t)res) ;
}

int64_t float_to_int(float f) {
  uint32_t s, e, m ;
  tie(s, e, m) = deconstruct_float(f) ;
  return float_to_int(s, e, m) ;
}

bool check(float f) {
  return float_to_int(f) == int64_t(f) ;
}

bool check(uint32_t s, uint32_t m, uint32_t e) {
  return check(construct_float(s, m, e)) ;
}

/*
CHECKPOINTS -
9223373136366403584.0 --> 0 10111110 00000000000000000000001
-9223373136366403584  --> 1 10111110 00000000000000000000001

*/

int main(int argc, char **argv) {
  cout.precision(23);

  // float f = 2147483648.8473598367 ;
  // uint32_t s, e, m ;
  // tie(s, e, m) = deconstruct_float(f) ;
  // cout << "The float " << f << " is (" << bitset<1>(s) << ", " << bitset<EBITS>(e) << ", " << bitset<MBITS>(m) << ")" << endl ;

  // uint32_t s, e, m ;
  // float f ;
  // s = 0b0 ;
  // e = 0b10111110 ;
  // m = 0b00000000000000000000001 ;
  // f = construct_float(s, e, m) ;
  // cout << "The (s, e, m) is (" << bitset<1>(s) << ", " << bitset<EBITS>(e) << ", " << bitset<MBITS>(m) << ")" ;
  // printf("and the float is %.23f\n", f) ;

  // int64_t i = float_to_int(f) ;
  // cout << "float_to_int() = " << i << " and int64_t() = " << int64_t(f) << endl ;

  bool correct = true ;

  for (uint32_t s = 1 ; s <= 1 ; s++) {
    for (uint32_t e = 0 ; e < (1 << EBITS) ; e++) {
      for (uint32_t m = 0 ; m < (1 << MBITS) ; m++) {
        float f = construct_float(s, e, m) ;

        if (isnormal(f) && (!check(s, e, m))) {
          printf("%.23f --> ", f) ;
          cout << s << " " << bitset<EBITS>(e) << " " << bitset<MBITS>(m) << endl ;
          correct = false ;
          exit(1) ;
        } 
      }
    }
  }

  return 0 ;
}


