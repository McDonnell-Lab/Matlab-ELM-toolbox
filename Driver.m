clear all;tic

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%this is an example script that implements an ELM network to classify images
%The main computation required is the calculation of output weights.
%Several methods for doijng this are provided.

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 1: load and prepare MNIST data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[X,X_test,labels,labels_test,ImageSize,NumClasses,k_train,k_test,Y,L] = PrepareMNISTData;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 1: set input weights, hidden layer, and output weights parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%note that parameters for specific algorithms are specified within the switch statement below.

%parameters for input weights
InputWeightFlags = [1,3,0]; %these default values are usually best for image classification
MinMaskSize = 10; %used only if InputWeightFlags(1) == 1
RF_Border = 3; %used only if InputWeightFlags(1) == 1
Scaling = 2;

%parameters for hidden units
M=1600;  %number of hidden units
HiddenUnitType = 'Relu'; %options: 'Relu', 'Sigmoid', 'Quadratic', 'Tanh', 'Relu','Cubic','SignedQuadratic'

%parameters for output weights: there are different choices of optimisation method
LearningMethod = 'RLS' %options: 'SingleBatchRidgeRegression', 'ConjGrad', 'Modular', 'RLS'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 2: get input layer weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W_input = GetInputLayerWeights(InputWeightFlags,L,ImageSize,X,Y,k_train,labels,NumClasses,M,MinMaskSize,RF_Border,Scaling);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 3: get Hidden Layer Activations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[A,A_test] = GetHiddenLayerActivations(W_input,X,X_test,HiddenUnitType);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 4: Compute the output weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
switch LearningMethod
    case 'SingleBatchRidgeRegression'
        %solve for the weights using least squares multilinear regression
        %Invention of the least squares method is disputed, with claims for
        %Gauss and for Legendre.
        %see S. M. Stigler (1981). "Gauss and the Invention of Least Squares". Ann. Stat. 9:465?47
        
        %Here we solve for output weights using Cholesky solver to solve linear equations
        %Another method is to use SVD, but this is much slower, so we don't use that
        Lambda = 1e-2; %ridge regression/weight decay regularisation parameter
        W_outputs = (A*Y)'/(A*A'+Lambda*eye(M));
    case 'ConjGrad'
        %The conjugate gradient method.
        
        MaxIterations = 100;
        ProgressFlag = 0;
        Lambda = 1e-8; %ridge regression/weight decay regularisation parameter
        W_outputs = ConjGrad_ELM(A,A_test,NumClasses,k_train,k_test,Y,labels,labels_test,Lambda,M,MaxIterations,ProgressFlag);
    case 'RLS'
        %Iteratively solve for the output weights using multiple batches of training samples and the Recursive-Least-Squares equations   
        
        BatchSize = 100; %seems to be fastest batchsize
        Lambda = 1e-2; % a larger Lambda seems necessary for larger batch sizes  
        [W_outputs,P] = RLS_ELM(A,Y,M,Lambda,BatchSize,NumClasses,k_train);
        
    case 'Modular'
        %Incrementally solve for the output weights using multiple batches of hidden units, i.e. modules.
        
        ModuleSize = 400;
        ProgressFlag =0;
        W_outputs = Modular_ELM(X,X_test,NumClasses,k_train,k_test,HiddenUnitType,W_input,Y,labels,labels_test,Lambda,M,ModuleSize,ProgressFlag);
 
    otherwise
        disp('No method selected')
        return
end
toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 4: verify classification on the training set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y_predicted_train = W_outputs*A;
[MaxVal,ClassificationID_train] = max(Y_predicted_train); %get output layer response and then classify it
PercentCorrect_train = 100*(1-length(find(ClassificationID_train-1-labels'~=0))/k_train) %calculate the error rate

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Step 5: get classification on test data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y_predicted_test = W_outputs*A_test;
[MaxVal,ClassificationID_test] = max(Y_predicted_test); %get output layer response and then classify it
PercentCorrect_test = 100*(1-length(find(ClassificationID_test-1-labels_test'~=0))/k_test)

