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

%Creamos un vector auxiliar con todo unos menos el primer elemento, que es 0, con el objetivo de vectorizar la operacion
% de regularizacion, ya que  multiplicamos la expresion de lambda/2*m por cada elemento de theta al cuadrado, y posteriormente
% con el vector auxiliar, para eliminar la regularizacion de theta0, la cual es uno por convencion y no necesita de
% regularizacion, de modo que al final al multiplicarse por el primer elemento del vector auxiliar que es 0, desaparece
% y no se regulariza, los demas si, ya que todos se multiplican por 1 y se suman a la regularizacion.
auxiliar_lambda = [0; ones(size(theta, 1)-1, size(theta, 2))];

J = 1/m*(-y'*log(sigmoid(theta'*X'))' - (1-y)'*log(1-sigmoid(theta'*X'))') + lambda/(2*m)*auxiliar_lambda'*theta.^2;

%En el gradiente, para añadir la regularizacion simplemente añadimos el termino de lambda/m a cada elemento de theta
% multiplicado por el vector auxiliar con la multiplicacion wise-element (.*), para asi eliminar del mismo modo que anteriormente
% el primer parametro theta0, que no necesita ser regularizado por convencion.
grad = 1/m*((sigmoid(theta'*X')'-y)'*X)' + (lambda/m)*(theta.*auxiliar_lambda);



% =============================================================

end
