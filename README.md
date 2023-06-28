# Codebase for inferring single nucleotide activation and repression maps using deep-learning

## How to use
input: data file, and target directory for the results

        A data file includes training, validation and test sets. 
        Each N-sample set is represented by:
           An N*1000*7 matrix:
           4 features of one-hot-encode DNA sequence matrix, followed by nucleotide-resolution signals for DNA methylation, H3K27ac and H3K4me1.
           An N*1000 target matrix: a per-position value that is a positive activation signal for enhancers, negative repression signal for silencers, and 0 otherwise.
           An N*3 target class matrix.

Download the data from: https://drive.google.com/file/d/16Zzf0OCCWdjcvUuwysWfMjvm9iPRZSjC/view?usp=sharing .
Decompress the file to 'data' folder.

Training a regression model:

        python train.deeptact.reg.py  ./data/data.hdf5 ./output/

output: in the directory ./output/
        
        mae.txt
        model_weights.reg.hdf5

Training a classification model:

        python train.deeptact.py  ./data/data.hdf5 ./output/

output: in the directory ./output/

        auc.txt
        model_weights.class.hdf5
