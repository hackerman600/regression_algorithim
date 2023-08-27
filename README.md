# regression_algorithim
Linear regression algorithim from scratch, 99% accuracy, it was indended to be easy and quick which is why the bias term and test dataset was excluded. 
I implemented the partial derivative algorithim from scratch demonstrating my understanding of calculus with respect to machine learning.

I am aware that the partial derivatives of mse (mean square error) with respect to each of the parameters involves using the chain rule. Mae in this case is a composite function consisting of another function being y_predict. 

the partial derivative of y_pred with respect to the weight matrix are the features. 
the partial derivative of mse with respect to y_predict = 2(y_pred - y)/n or 2(absolute_error) / n.
this leaves us with: 2 * x.T * absolute_error / n. 
