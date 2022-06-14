printf "4, 4; 4\n"
./gcnlstm-fixed r=2 mbits=7 ebits=8 nt=16 sz1=4 sz2=4

printf "\n4, 4; 16\n"
./gcnlstm-fixed r=2 mbits=7 ebits=8 nt=16 sz1=4 sz2=16

printf "\n16, 16; 4\n"
./gcnlstm-fixed r=2 mbits=7 ebits=8 nt=16 sz1=16 sz2=4

printf "\n32, 32; 4\n"
./gcnlstm-fixed r=2 mbits=7 ebits=8 nt=16 sz1=32 sz2=32

printf "\n4, 4; 8\n"
./gcnlstm-fixed r=2 mbits=7 ebits=8 nt=16 sz1=4 sz2=8

printf "\n8, 8; 4\n"
./gcnlstm-fixed r=2 mbits=7 ebits=8 nt=16 sz1=8 sz2=4

printf "\n4, 4; 32\n"
./gcnlstm-fixed r=2 mbits=7 ebits=8 nt=16 sz1=4 sz2=32
