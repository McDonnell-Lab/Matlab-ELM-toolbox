function [A,A_test] = GetHiddenLayerActivations(W_input,X,X_test,HiddenUnitType)

switch HiddenUnitType
    case 'Relu'
        A = max(0,W_input*X);
        A_test = max(0,W_input*X_test);
    case 'Sigmoid'
        A = 1./(1+exp(-W_input*X));
        A_test =1./(1+exp(-W_input*X_test));
    case 'Tanh'
        A = tanh(W_input*X);
        A_test = tanh(W_input*X_test);
    case 'Quadratic'
        A = (W_input*X).^2;
        A_test = (W_input*X_test).^2;
    case 'Cubic'
        A = (W_input*X).^3;
        A_test = (W_input*X_test).^3;
    case 'SignedQuadratic'
        A = sign(W_input*X).*(W_input*X).^2;
        A_test = sign(W_input*X_test).*(W_input*X_test).^2;
end
A = double(A);