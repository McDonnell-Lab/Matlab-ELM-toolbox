function [W_outputs,Y_predicted,Y_predicted_test] = Modular_ELM(X,X_test,NumClasses,k_train,k_test,HiddenUnitType,W_input,Y,labels,labels_test,Lambda,M,ModuleSize,ProgressFlag,StoppingValue)
%
% This function calculates an approximate incremental solution to the output weights.
%
% This method was described in
%    M. D. Tissera and M. D. McDonnell.
%    "Modular Expansion of the Hidden Layer in Single Layer
%    Feedforward Neural Networks."
%    Proc. IJCNN, Vancouver, 2016.
%
% The method extends beyond increments of single nodes, which is the I-ELM method of
%   G.-B. Huang, L. Chen and C.-K. Siew
%   "Universal Approximation Using Incremental Constructive Feedforward Networks With Random Hidden Nodes"
%   IEEE Transactions on Neural Networks, 17:879-892, 2006
%
% This code requires implicit inversion of matrices of size ModuleSize x ModuleSize
% it also requires explicit multiplication of (ModuleSize x k_train)(k_train x ModuleSize) matrices

W_outputs = zeros(NumClasses,M);
Y_predicted = zeros(NumClasses,k_train);
Y_predicted_test = zeros(NumClasses,k_test);
dMSE = 1e10;
Err = Y';
MSE = mean(mean(Err.^2));
tic
for m = 1:floor(M/ModuleSize)
    MSE_prev = MSE;
    
    %update
    [A,A_test] = GetHiddenLayerActivations(W_input((m-1)*ModuleSize+1:m*ModuleSize,:),X,X_test,HiddenUnitType);
    W_outputs_m =  single((Err*A')/(A*A'+Lambda*eye(ModuleSize)));
    Y_predicted = Y_predicted + W_outputs_m*A;
    Y_predicted_test = Y_predicted_test + W_outputs_m*A_test;
   
    W_outputs(:,(m-1)*ModuleSize+1:m*ModuleSize) =  W_outputs_m;
    
    Err = Y'-double(Y_predicted);
    MSE = mean(mean(Err.^2))
    
    if dMSE < StoppingValue
        disp('MSE reached target stopping value; returning')
        return
    end
    
    dMSE = MSE_prev-MSE;
    
    if ProgressFlag
        
        %evaluations
        disp(['Difference in MSE after iteration ' num2str(m) ': ' num2str(dMSE)])

        [~,ClassificationID_train] = max(Y_predicted); %get output layer response and then classify it
        PercentCorrect_train = 100*(1-length(find(ClassificationID_train-1-labels'~=0))/k_train) %calculate the error rate
        
        [~,ClassificationID_test] = max(Y_predicted_test); %get output layer response and then classify it
        PercentCorrect_test = 100*(1-length(find(ClassificationID_test-1-labels_test'~=0))/k_test)
        toc
    end
    
    
end

