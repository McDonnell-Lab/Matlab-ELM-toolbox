Author: A/Prof Mark D. McDonnell, University of South Australia
Email: mark.mcdonnell@unisa.edu.au
Date: April 2016

Citation: If you find this code useful, please read and cite the following papers:

[1] M. D. McDonnell, M. D. Tissera, T. Vladusich, A. van Schaik and J. Tapson.
   Fast, simple and accurate handwritten digit classification by training shallow neural network classifiers
   with the "extreme learning machine" algorithm. PLOS One, 10: Article Number e0134254, 2015.

[2] M. D. Tissera and M. D. McDonnell.
   Modular Expansion of the Hidden Layer in Single Layer Feedforward Neural Networks. 
   Accepted for Proc. IJCNN, Vancouver, 2016.

This toolbox provides code useful for implementing computation of least squares optimal computation of weights matrices, 
such as for application in single-hidden layer neural networks with a linear output layer, applied to
classifying images, as in refs [1] and [2] above.

The main computation required is the optimisation of output weights to minimise mean square error between training targets and data.

Several methods for doing this are provided. Several example scripts are also provided, showing application to 
classification of the standard MNIST image database.

1. The generally simplest method is to solve for all training data in a
single batch using linear equations using a Cholesky solver.

2. The main deficiency in the single-batch Cholesky method is its high memory burden
   for large hidden layers. Four methods here can circumvent this: the
   conjugate gradient method, the modular method,  the epsilon-NLMS
   method and the `Big Data' method.
   
3. The standard Cholesky method also is not good for non-stationary data. The RLS
   and epsilon-NLMS methods are better for non-stationary data, as in the OPIUM and OPIUM-light methods (see comments in E_NLMS.m).
   
4. The modular method offers several advantages: less memory intensive,
   and also faster than the standard method for the same total number of
   hidden units. However it loses some accuracy and generally a higher number of hidden units is
   necessary to get close to the same performance.
   
5.  The 'Big Data' method simply iteratively computes the Gram matrix for
   use in the standard Cholesky solver approach.
