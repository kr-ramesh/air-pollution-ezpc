printf "FloatML 7 Dec\n"
cat ../7Dec/inputs2.txt ../7Dec/A.inp ../7Dec/kernel.inp ../7Dec/bias.inp ../7Dec/A2.inp ../7Dec/kernel2.inp ../7Dec/bias2.inp ../7Dec/kernel3.inp ../7Dec/recurrent_kernel3.inp ../7Dec/bias3.inp ../7Dec/kernel4.inp ../7Dec/bias4.inp ../7Dec/labels2.txt | ./gcnlstm-cin-floatml r=1 nt=16

printf "FloatML 8 Dec\n"
cat ../8Dec/inputs2.txt ../8Dec/A.inp ../8Dec/kernel.inp ../8Dec/bias.inp ../8Dec/A2.inp ../8Dec/kernel2.inp ../8Dec/bias2.inp  ../8Dec/kernel3.inp ../8Dec/recurrent_kernel3.inp ../8Dec/bias3.inp ../8Dec/kernel4.inp ../8Dec/bias4.inp ../8Dec/labels2.txt | ./gcnlstm-cin-floatml r=1 nt=16

printf "FloatML 9 Dec\n"
cat ../9Dec/inputs2.txt ../9Dec/A.inp ../9Dec/kernel.inp ../9Dec/bias.inp ../9Dec/A2.inp ../9Dec/kernel2.inp ../9Dec/bias2.inp ../9Dec/kernel3.inp ../9Dec/recurrent_kernel3.inp ../9Dec/bias3.inp ../9Dec/kernel4.inp ../9Dec/bias4.inp ../9Dec/labels2.txt | ./gcnlstm-cin-floatml r=1 nt=16

printf "FloatML 10 Dec\n"
cat ../10Dec/inputs2.txt ../10Dec/A.inp ../10Dec/kernel.inp ../10Dec/bias.inp ../10Dec/A2.inp ../10Dec/kernel2.inp ../10Dec/bias2.inp ../10Dec/kernel3.inp ../10Dec/recurrent_kernel3.inp ../10Dec/bias3.inp ../10Dec/kernel4.inp ../10Dec/bias4.inp ../10Dec/labels2.txt | ./gcnlstm-cin-floatml r=1 nt=16

printf "FloatML 11 Dec\n"
cat ../11Dec/inputs2.txt ../11Dec/A.inp ../11Dec/kernel.inp ../11Dec/bias.inp ../11Dec/A2.inp ../11Dec/kernel2.inp ../11Dec/bias2.inp ../11Dec/kernel3.inp ../11Dec/recurrent_kernel3.inp ../11Dec/bias3.inp ../11Dec/kernel4.inp ../11Dec/bias4.inp ../11Dec/labels2.txt | ./gcnlstm-cin-floatml r=1 nt=16 

