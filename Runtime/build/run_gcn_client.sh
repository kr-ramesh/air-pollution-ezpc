printf "FloatML FP32\n"
./gcnlstm-nocin-floatml r=2 nt=16 add=10.0.0.4

printf "FloatML BF16\n"
./gcnlstm-nocin-floatml r=2 nt=16 mbits=7 ebits=8 add=10.0.0.4

printf "Beacon FP32\n"
./gcnlstm-nocin-beacon r=2 nt=16 add=10.0.0.4

printf "Beacon FP16\n"
./gcnlstm-nocin-beacon r=2 nt=16 mbits=7 ebits=8 add=10.0.0.4
