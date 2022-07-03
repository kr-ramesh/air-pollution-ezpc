printf "FloatML FP32\n"
./gcnlstm-nocin-floatml r=1 nt=16

printf "FloatML BF16\n"
./gcnlstm-nocin-floatml r=1 nt=16 mbits=7 ebits=8

printf "Beacon FP32\n"
./gcnlstm-nocin-beacon r=1 nt=16

printf "Beacon FP16\n"
./gcnlstm-nocin-beacon r=1 nt=16 mbits=7 ebits=8
