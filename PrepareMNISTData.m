function [X,X_test,labels,labels_test,ImageSize,NumClasses,k_train,k_test,Y,L] = PrepareMNISTData()

%get training and test data; data files can be obtained from http://yann.lecun.com/exdb/mnist/
X = loadMNISTImages('train-images-idx3-ubyte'); %uses function available from http://ufldl.stanford.edu/wiki/resources/mnistHelper.zip
X_test = loadMNISTImages('t10k-images-idx3-ubyte'); %uses function available from http://ufldl.stanford.edu/wiki/resources/mnistHelper.zip
labels = loadMNISTLabels('train-labels-idx1-ubyte'); %uses function available from http://ufldl.stanford.edu/wiki/resources/mnistHelper.zip
labels_test = loadMNISTLabels('t10k-labels-idx1-ubyte'); %uses function available from http://ufldl.stanford.edu/wiki/resources/mnistHelper.zip

%create a target matrix using the training class labels
ImageSize = 28;
NumClasses = 10;
k_train = 60000; %number of training samples 
k_test = 10000; %number of test samples
Y = zeros(length(labels),NumClasses); %one-hot target matrix
for ii = 1:length(labels)
    Y(ii,labels(ii)+1) = 1;
end

%assume there may be a bias. It can be set to zero by setting the input
%weights to zero
X = single([X;ones(1,k_train)]);
X_test = single([X_test;ones(1,k_test)]);
L=ImageSize^2+1; %dimension of each sample