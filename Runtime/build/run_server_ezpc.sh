printf "FloatML 7 Dec\n"
cat ../7Dec/inputs2.txt ../7Dec/A.inp ../7Dec/kernel.inp ../7Dec/bias.inp ../7Dec/A2.inp ../7Dec/kernel2.inp ../7Dec/bias2.inp ../7Dec/kernel3.inp ../7Dec/reck.inp ../7Dec/bias3.inp ../7Dec/kernel4.inp ../7Dec/bias4.inp ../7Dec/labels2.txt | ./gcnlstm-cin-floatml r=1 nt=16

printf "FloatML 8 Dec\n"
cat ../8Dec/inputs2.txt ../Weights/A1.txt ../Weights/kernel1.txt ../Weights/bias1.txt ../Weights/A2.txt ../Weights/kernel2.txt ../Weights/bias2.txt ../Weights/k.txt ../Weights/reck.txt ../Weights/lstmbias.txt ../Weights/dense.txt ../Weights/bias4.txt ../8Dec/labels2.txt | ./gcnlstm-cin-floatml r=1 nt=16

printf "FloatML 9 Dec\n"
cat ../9Dec/inputs2.txt ../Weights/A1.txt ../Weights/kernel1.txt ../Weights/bias1.txt ../Weights/A2.txt ../Weights/kernel2.txt ../Weights/bias2.txt  ../Weights/k.txt ../Weights/reck.txt ../Weights/lstmbias.txt ../Weights/dense.txt ../Weights/bias4.txt ../9Dec/labels2.txt | ./gcnlstm-cin-floatml r=1 nt=16

printf "FloatML 10 Dec\n"
cat ../10Dec/inputs2.txt ../Weights/A1.txt ../Weights/kernel1.txt ../Weights/bias1.txt ../Weights/A2.txt ../Weights/kernel2.txt ../Weights/bias2.txt  ../Weights/k.txt ../Weights/reck.txt ../Weights/bias3.txt ../Weights/dense.txt ../Weights/bias4.txt ../10Dec/labels2.txt | ./gcnlstm-cin-floatml r=1 nt=16

printf "FloatML 16 Dec\n"
cat ../11Dec/inputs2.txt ../Weights/A1.txt ../Weights/kernel1.txt ../Weights/bias1.txt ../Weights/A2.txt ../Weights/kernel2.txt ../Weights/bias2.txt  ../Weights/k.txt ../Weights/reck.txt ../Weights/bias3.txt ../Weights/dense.txt ../Weights/bias4.txt ../11Dec/labels2.txt | ./gcnlstm-cin-floatml r=1 nt=16

