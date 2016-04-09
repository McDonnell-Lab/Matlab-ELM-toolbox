clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%This is an example script that implements an ELM network to classify MNIST images at well over 99.1% on the test set.
%It uses the method of ref [2] below.

%Author: A/Prof Mark D. McDonnell, University of South Australia
%Email: mark.mcdonnell@unisa.edu.au
%Date: April 2016

%Citation: If you find this code useful, please read and cite the following papers:

%[1] M. D. McDonnell, M. D. Tissera, T. Vladusich, A. van Schaik and J. Tapson.
%    Fast, simple and accurate handwritten digit classification by training shallow neural network classifiers
%    with the "extreme learning machine" algorithm. PLOS One, 10: Article Number e0134254, 2015.

%[2] M. D. Tissera and M. D. McDonnell.
%    Modular Expansion of the Hidden Layer in Single Layer Feedforward Neural Networks.
%    Accepted for Proc. IJCNN, Vancouver, 2016.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[X,X_test,labels,labels_test,ImageSize,NumClasses,k_train,k_test,Y,L] = PrepareMNISTData;
InputWeightFlags = [1,3,0]; %these default values are usually best for image classification
MinMaskSize = 10; %used only if InputWeightFlags(1) == 1
RF_Border = 0; %used only if InputWeightFlags(1) == 1
M=200000;  %number of hidden units
HiddenUnitType = 'Relu'; %options: 'Relu', 'Sigmoid', 'Quadratic', 'Tanh', 'Relu','Cubic','SignedQuadratic'
LearningMethod = 'Modular' %options: 'SingleBatchRidgeRegression', 'ConjGrad', 'Modular', 'RLS', 'e-NLMS'
ModuleSize = 5000;
ProgressFlag =1;
Lambda = 1e-3;
Scaling = 2;
StoppingValue = 1e-4;

tic;W_input = GetInputLayerWeights(InputWeightFlags,L,ImageSize,X,Y,k_train,labels,NumClasses,M,MinMaskSize,RF_Border,Scaling);toc
[W_outputs,Y_predicted_train,Y_predicted_test] = Modular_Regression(X,X_test,NumClasses,k_train,k_test,HiddenUnitType,W_input,Y,labels,labels_test,Lambda,M,ModuleSize,ProgressFlag,StoppingValue);

