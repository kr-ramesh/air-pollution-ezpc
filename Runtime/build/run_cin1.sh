printf "Beacon\n"
./gcnlstm-cin-beacon r=2 mbits=7 ebits=8 add=10.0.0.4 nt=16

printf "FloatML BF16\n"
./gcnlstm-cin-floatml r=2 mbits=7 ebits=8 add=10.0.0.4 nt=16

# printf "FloatML FP32\n"
# ./gcnlstm-cin-floatml r=2 add=10.0.0.4 nt=16
