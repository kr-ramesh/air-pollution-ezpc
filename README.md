# air-pollution-ezpc

The Notebooks folder contains the PyTorch and Keras equivalents of the GCN-LSTM model. I've hand-written the equations in the PyTorch notebooks for as I used them for reference while debugging the EzPC model.
The C++ implementation of the GCN-LSTM is stored within the gcn_implementations_c++ folder. The ezpc code/functions folder contains the code for the inference functions which was compiled using the EMP backend.

The 'GCN in PyTorch' notebook contains the implementation of the GCN-LSTM in PyTorch, with a single LSTM layer. This would come in handy for understanding the architecture of the GCN-LSTM.
gcnlstm_training_sec_final.cpp is the file to refer to for the complete working implementation of the GCN-LSTM in C++. I've written separate functions for the Fixed Adjacency GCN layer and the LSTM, which are subsequently used in the forward pass. The backpropagation function within the LSTM (void backward()) computes the gradients for all the weights.
