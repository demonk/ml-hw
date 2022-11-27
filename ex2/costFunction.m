function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% 需要计算两个值，一个是J，一个是grad，分开计

% H = g(z)
%theta=theta[:,np.newaxis] %给列增加theta维度
h=sigmoid(X*theta) %实现在sigmod.m

% J 的表达式参考笔记
J=(1/m)*((-y)'*log(h)-(1-y)'*log(1-h));

% 梯度即为J的微分
diff_hy=h-y;
grad = (1/m)*(X'*diff_hy);

%return J,grad
% =============================================================

end
