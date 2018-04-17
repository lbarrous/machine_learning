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

% Definimos la funcion de coste en la cual hacemos la traspuesta de y para multiplicarla por su correspondiente
% valor de la hipotesis (La cual es el valor de la funcion sigmoide del producto entre cada valor de theta con todos
% sus correspondientes valores de esa caracteristica en X, de tal modo que conseguimos un vector columna que tendra
% 1xm, y que tendra el valor de la hipotesis para cada ejemplo de entrenamiento menos el valor de su correspondiente y

J = 1/m*(-y'*log(sigmoid(theta'*X'))' - (1-y)'*log(1-sigmoid(theta'*X'))');

grad = 1/m*((sigmoid(theta'*X')'-y)'*X);





% =============================================================

end
