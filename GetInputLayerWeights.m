function [W_randoms] = GetInputLayerWeights(Flags,L,ImageSize,X,Y,k_train,labels,NumClasses,M,MinMaskSize,RF_Border,Scaling)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This code implements methods described and referenced in the paper:

% M. D. McDonnell, M. D. Tissera, T. Vladusich, A. van Schaik and J. Tapson. 
% Fast, simple and accurate handwritten digit classification by training shallow neural network classifiers 
% with the "extreme learning machine" algorithm. PLOS One, 10: Article Number e0134254, 2015.

% Author: Assoc Prof Mark D. McDonnell, University of South Australia
% Email: mark.mcdonnell@unisa.edu.au
% Date: January 2015
% Citation: If you find this code useful, please cite the paper:
% M. D. McDonnell, M. D. Tissera, T. Vladusich, A. van Schaik and J. Tapson. 
% Fast, simple and accurate handwritten digit classification by training shallow neural network classifiers 
% with the "extreme learning machine" algorithm. PLOS One, 10: Article Number e0134254, 2015.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Mask = ones(M,L-1,'single');
if Flags(1) == 1
    %get receptive field masks
    Mask =  zeros(M,L-1);
    for ii = 1:M
        SquareMask = zeros(ImageSize,ImageSize);
        Inds1 = zeros(2,1);Inds2 = zeros(2,1);
        while (Inds1(2)-Inds1(1))*(Inds2(2)-Inds2(1)) < MinMaskSize
            Inds1 = RF_Border+sort(randperm(ImageSize-2*RF_Border,2));
            Inds2 = RF_Border+sort(randperm(ImageSize-2*RF_Border,2));
        end
        SquareMask(Inds1(1):Inds1(2),Inds2(1):Inds2(2))=1;
        Mask(ii,:) =  SquareMask(:);
    end
end

W_randoms = zeros(M,L-1,'single');
biases = zeros(M,1,'single');
switch Flags(2)
    case 1
        %get random weights
        W_randoms = single(sign(randn(M,L-1))); %get bipolar random weights
        W_randoms =  Mask.*W_randoms; %mask random weights
        W_randoms = Scaling*diag(1./sqrt(eps+sum(W_randoms.^2,2)))*W_randoms; %normalise rows and scale
        biases = 0.1*single(randn(M,1));
    case 2
        %get CIW or RF-CIW weights
        NumEachClass = sum(Y);
        M_CIWs = round(M*NumEachClass/k_train);
        M0 = sum(M_CIWs);
        if M0 ~= M
           M_CIWs(1) = M_CIWs(1) + M - M0;
        end
        Count = 1;
        for i = 1:NumClasses
            ClassIndices = find(labels==i-1);
            W_randoms(Count:Count+M_CIWs(i)-1,:) = single(sign(randn(M_CIWs(i),length(ClassIndices)))*X(:,ClassIndices)');
            Count = Count + M_CIWs(i);
        end
        W_randoms =  Mask.*W_randoms; %mask random weights
        W_randoms = Scaling*diag(1./sqrt(eps+sum(W_randoms.^2,2)))*W_randoms; %normalise rows and scale
    case 3
        %Get the Constrained (C) weights or RF-C weights
        for i = 1:M
            Norm = 0;
            Inds = ones(2,1);
            while  Norm < eps
                Inds = randperm(k_train,2);
                X_Diff = single(X(1:L-1,Inds(1))-X(1:L-1,Inds(2)));
                Wrow = X_Diff.*Mask(i,:)'; %get masked C random weights
                Wrow = Wrow-mean(Wrow);
                Norm  = sqrt(sum(Wrow.^2));
            end
            W_randoms(i,:) = Wrow/Norm;
            biases(i) = 0.5*(single(X(1:L-1,Inds(1))+X(1:L-1,Inds(2))))'*Wrow/Norm; 
        end
        W_randoms = Scaling*W_randoms; %scale the weights (already normalised) 
    otherwise
        disp('error')
        return
end
if Flags(3) == 1
    W_randoms = [W_randoms biases];
else
    W_randoms = [W_randoms single(zeros(size(biases)))];
end