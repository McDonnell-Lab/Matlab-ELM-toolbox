function [W_outputs] = ConjGrad_ELM(A,A_test,NumClasses,k_train,k_test,Y,labels,labels_test,Lambda,M,MaxIts,ProgressFlag)

%See M. Hestenes and E. Stiefel (1952). "Methods of Conjugate Gradients for Solving Linear Systems".
%Journal of Research of the National Bureau of Standards 49 (6).

%Here we apply the Conjugate Gradient method to the normal equations.
%See C. T. Kelley, Iterative Methods for Linear and Nonlinear Equations. SIAM, 1995.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%init CG method applied to the normal equations.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MSE = zeros(MaxIts,1);
W_outputs = zeros(NumClasses,M);
b=(A*Y)'/k_train; %(N x M) - initialise to the mean in each class
r_k = b;
p_k = r_k;

tic
for Iteration = 1:MaxIts
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %iterate CG method applied to the normal equations.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %these two lines are what enables us to avoid calculating an (M x K)(K x M) matrix multiplications
    %we never need to create an M x M matrix, so M can be very large
    
    %we divide by k_train to ensure each entry of the Gram matrix is an average over all samples
    h = p_k*A/sqrt(k_train); %(N x K)
    f_k = A*h'/sqrt(k_train); %(N x M)
    
    %ridge regression regularization:
    f_k = f_k' + Lambda*p_k;
    
    %update the weights
    d_k = sum(r_k(:).^2);
    alpha_k = d_k/sum(h(:).^2);
    W_outputs = W_outputs + alpha_k*p_k;
    Y_predicted = W_outputs*A;
    
    r_k = r_k-alpha_k*f_k;
    beta_k = sum(r_k(:).^2)/d_k;
    p_k = r_k + beta_k*p_k;
    
    MSE(Iteration) = mean(mean((Y'-Y_predicted).^2));
    if Iteration>1 && MSE(Iteration-1) - MSE(Iteration) < 1e-15
        disp(['MSE converged to 1e-6 after ' num2str(Iteration) ' iterations'])
        break
    end
    
    if ProgressFlag
        %performance evaluation
        disp(['Iteration: ' num2str(Iteration)])
        
        [~,ClassificationID_train] = max(Y_predicted);%get output layer response and then classify it
        PercentCorrect_train = 100*(1-length(find(ClassificationID_train-1-labels'~=0))/k_train)
        
        [~,ClassificationID_test] = max(W_outputs*A_test); %get classification
        PercentCorrect_test = 100*(1-length(find(ClassificationID_test-1-labels_test'~=0))/k_test)
        
        disp(['MSE: ' num2str(MSE(Iteration))])
    end
end
