function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


thetax=X*theta;
gthetax=sigmoid(thetax);
eachcost=0-(y.*log(gthetax) + (1-y).*log(1-gthetax));
cost=(sum(eachcost))/m;

reg_factor=lambda/(2*m) * (sum(theta.*theta) - theta(1,1)*theta(1,1));
J=cost+reg_factor;

theta1=(gthetax-y).*X(:,1);
theta2=(gthetax-y).*X(:,2);
theta3=(gthetax-y).*X(:,3);


grad(1,1)=(sum(theta1))/m
grad(2,1)=(sum(theta2))/m + lambda*theta(2,1)/m;
grad(3,1)=(sum(theta3))/m + lambda*theta(3,1)/m;



% =============================================================

end
