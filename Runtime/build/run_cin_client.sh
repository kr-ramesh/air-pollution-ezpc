printf "FloatML 7 Dec\n"
cat ../7Dec/inputs2.txt ../7Dec/A.inp ../7Dec/kernel.inp ../7Dec/bias.inp ../7Dec/A2.inp ../7Dec/kernel2.inp ../7Dec/bias2.inp ../7Dec/kernel3.inp ../7Dec/recurrent_kernel3.inp ../7Dec/bias3.inp ../7Dec/kernel4.inp ../7Dec/bias4.inp ../7Dec/labels2.txt | ./gcnlstm-cin-floatml r=1 nt=16

