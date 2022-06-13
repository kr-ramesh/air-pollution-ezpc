printf "Beacon\n"
cat ../../files/inputs2.txt ../../files/gcn_layer1.A.txt ../../files/gcn_layer1.kernel.txt ../../files/gcn_layer1.bias.txt ../../files/gcn_layer2.A.txt ../../files/gcn_layer2.kernel.txt ../../files/gcn_layer2.bias.txt ../../files/hidden.txt ../../files/cell.txt  ../../files/lstm.weight_ih_l0.txt ../../files/lstm.weight_hh_l0.txt ../../files/lstmbias.txt ../../files/dense.weight.txt ../../files/dense.bias.txt ../../files/labels2.txt | ./gcnlstm-cin-beacon r=1 mbits=7 ebits=8 nt=16
# ./gcnlstm-beacon r=1 mbits=7 ebits=8 nt=16

printf "FloatML\n"
cat ../../files/inputs2.txt ../../files/gcn_layer1.A.txt ../../files/gcn_layer1.kernel.txt ../../files/gcn_layer1.bias.txt ../../files/gcn_layer2.A.txt ../../files/gcn_layer2.kernel.txt ../../files/gcn_layer2.bias.txt ../../files/hidden.txt ../../files/cell.txt  ../../files/lstm.weight_ih_l0.txt ../../files/lstm.weight_hh_l0.txt ../../files/lstmbias.txt ../../files/dense.weight.txt ../../files/dense.bias.txt ../../files/labels2.txt | ./gcnlstm-cin-floatml r=1 mbits=7 ebits=8 nt=16
# ./gcnlstm-floatml r=1 mbits=7 ebits=8 nt=16
