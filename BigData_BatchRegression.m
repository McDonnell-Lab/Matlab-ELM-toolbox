function [W_outputs] = BigData_ELM(A,Y,M,Lambda, BatchSize,NumClasses,k_train)
%
%This code is useful if its not possible to store all the data in memory at a single time.
%Users can adapt this code so that only batches of data are loaded into
%memory, and the Gram matrix A*A' is incrementally updated.
%The computation of the output weights, W_outputs = Q/G; is relatively fast
%and less memory-intensive.
%
%Citation: If you find this code useful, please read and cite the following paper that describes this method:
%
%[1] M. D. McDonnell, M. D. Tissera, T. Vladusich, A. van Schaik and J. Tapson.
%    Fast, simple and accurate handwritten digit classification by training shallow neural network classifiers
%    with the "extreme learning machine" algorithm. PLOS One, 10: Article Number e0134254, 2015.
%
%For single sample updates, Tte method alsos appear in the Kalman filter literature under the name of "Information filter"
%
%This function avoids a single (Mxk_train)(k_trainxM) computation of the Gram matrix, A*A'
%by using iterations over batches of training data

W_outputs = zeros(NumClasses,M);
G = Lambda*eye(M);
Q = zeros(NumClasses,M);
for kk = 1:floor(k_train/BatchSize)
    a = A(:,(kk-1)*BatchSize+1:kk*BatchSize);
    y = Y((kk-1)*BatchSize+1:kk*BatchSize,:);
    G = G + a*a';
    Q = Q + (a*y)';
end
W_outputs = Q/G;
