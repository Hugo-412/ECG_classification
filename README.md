# stage_ecg
This repository purpose is to provide a code for the study of Electrocardiogramms signals. We use Convolutional Dictionary Learning as well as other signal representations: raw data, signal peaks, a set of basic signal features, and special indicators based on the wavelet transform and autoregression. Then we apply a set of classifier from scikit-learn to evaluate the representations performances.
The Atomes.ipynb file contains the necessary code to perform the CDL on the ECGs signals.
The Classifier.ipynb file contains the code used to calculate the other representations and comper all of them using error rate, precision, recall and F1-score
The Wavelet.ipynb file is used to calculate the special indicators and their performances
The ECGData.mat file contain the ECGs datas used throughout this study
The Atome.mat file contains the result of a pre-executed Atomes.ipynb code

If you want to use this code we strongly advise use to read the  Etude comparatives des methodes de classification ECG.pdf wich is the corresponding study of this code and contains essential information for comprehension. We provide an english version of this study. However, note that it is only an AI translation of the french version and it has not been verified by man therefore we do not guaranty its accuracy.
Also we advise you to keep the repository architecture to avoid any execution problem.

Please feel free to download, use, complete or share this code any return will be appreciated.
If you use this code in your project, please consider citing our work : 
Fayadas H.,(2026).Etude comparatives des methodes de classification ECG
