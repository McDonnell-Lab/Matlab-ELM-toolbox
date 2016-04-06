function [W_outputs,P] = RLS_ELM(A,Y,M,Lambda, BatchSize,NumClasses,k_train)
%
% The Recursive Least Squares equations have been attributed to Gauss.
% In 1950 they were rediscovered in R.L.Plackett, Some Theorems in Least Squares, Biometrika, 1950, 37, 149-157
% They are an integral component of Kalman filters invented in the 1950s and 1960s.
%
% In the context of ELMs, the equations were re-derived in
% N.-Y. Liang, G.-B. Huang,  P. Saratchandran and N. Sundararajan,
% "A fast and accurate online sequential learning algorithm for feedforward networks",
% IEEE Transactions on Neural Networks", 17:1411-1423, 2006
%
% They also appear as part of the "OPIUM" method of
% J. Tapson and A. van Schaik, "Learning the pseudoinverse solution to network weights",
% Neural Networks, 45:94-100, 2013
%
% This function calculates an exact iterative solution to the optimal least squares output weights, if Lambda = 1.
% 
% The RLS equations are used in Kalman filters, and K is there called the Kalman gain.
% A simple way to derive this result is to start with the Woodbury matrix identity
% applied to calculating the inverse of the Gram matrix.
%
% This code requires implicit inversion of matrices of size BatchSize x Batchsize.
% It also requires explicit multiplication of (M x BatchSize)(BatchSize x M) matrices.
%
%set Lambda < 1 for non-stationary data

P = eye(M);
W_outputs = zeros(NumClasses,M);
for kk = 1:floor(k_train/BatchSize)
    
    %data and target
    a = A(:,(kk-1)*BatchSize+1:kk*BatchSize);
    T = Y((kk-1)*BatchSize+1:kk*BatchSize,:)';
    Err = T - W_outputs*a;
    
    %RLS equations
    r = P*a;
    K = r/(a'*r + Lambda*eye(BatchSize));
    P = P - r*K';
    W_outputs = W_outputs + Err*K';
    
end
