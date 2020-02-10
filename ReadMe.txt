Tabs in Data.xlsx:
	Data - Original: data in its original form
	Clean - Original: data cleaned ready to be tested
	Clean - New: data with new geographic labels ready to be tested, original data had too many
 	features

All normalization used was min-max normalization.

sharktank.m is the main matlab file. Notes inside provide greater clarity. I saved figures from this file in the folder 'Figures'. getaccuracy.m, ridge_regression.m, TestData.mat, and TrainData.mat are only used in sharktank.m.

Figures not found in file were created in tableau.

Accuracies in sharktank.m:

ridge_accuracy: 
    0.5938

libsvm_accuracy: 
    0.6062

nn_accuracy: 
    0.6250

non_linear_least_squares_accuracy: 
    0.6125

our_model_accuracy: 
    0.6250

knn_accuracy: 
    0.6062

