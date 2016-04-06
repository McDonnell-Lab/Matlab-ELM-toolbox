function [W_outputs,P] = E_NLMS_ELM(A,Y,M,Lambda, BatchSize,NumClasses,k_train,Delta,Runs)

%The normalised variation of the LMS algorithm, invented by Widrow.
%See for example D. P. Mandic, S. Kanna and A. G. Constantinides,
%On the intrinsic relationship between the least mean square and {K}alman filters,
%IEEE Signal Processing Magazine, 32:117-122, 2015

W_outputs = zeros(NumClasses,M);

for rr = 1:Runs
    for kk = 1:floor(k_train/BatchSize)
        
        %data and target
        a = A(:,(kk-1)*BatchSize+1:kk*BatchSize);
        T = Y((kk-1)*BatchSize+1:kk*BatchSize,:)';
        Err = T - W_outputs*a;
        
        %Normalised LMS equations
        K = a/(a'*a + Lambda*eye(BatchSize));
        W_outputs = W_outputs + Delta*Err*K';
        
    end
end


   