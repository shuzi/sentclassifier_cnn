CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ~/torch_openmpi/install/bin/th cnn.lua -batchSize 200 -type cuda -batchSizeTest 500 -trainFile ~/yelp/train_500k -validFile ~/yelp/valid_2000 -testFile ~/yelp/test_2000 -embeddingFile ~/yelp/filtered.vectors100.txt -numLabels 5 -trainMaxLength 150 -testMaxLength 300