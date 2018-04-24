function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% (Theta traspuesta * X traspuesta) traspuesta nos dara el valor de cada elemento de la hipotesis, es decir, un
% vector 1xm en el cual tendremos en cada elemento el valor de la regresion lineal (sumatorio de los productos de cada parametro por cada elemento del ejemplo de entrenamiento),
% si hacemos la traspuesta de esa matriz podremos restar a cada elemento de la hipotesis su correspondiente valor y

sumatorio = (theta'*X')'-y;

% Elevar al cuadrado cada elemento del resultado anterior
sumatorio_al_cuadrado = sumatorio.^2;

% Finalmente sumamos cada elemento del vector mx1 que tenemos
sumatorio_total = sum(sumatorio_al_cuadrado);

J = (1/(2*m))*sumatorio_total;

%Para la regularizacion, con el objetivo de evitar regulizar theta(1) que corresponde al primero termino de theta que nonzeros
% se regulariza, multiplicamos theta por unvector que tiene 0 en el primer elemento y 1 en todos los demas, para asi
% eliminar el primer parametro theta, y dejar los siguientes sin modificar, para asi seguidamente, elevar cada elemento
% al cuadrado y luego realizar la suma de todos y obtener finalmente el termino de regularizacion
regularized = sum(([zeros(1, size(theta,2)); ones(size(theta,1)-1, size(theta,2))] .* theta).^2);

J = J + (lambda/(2*m))*regularized;

%Con el gradiente hacemos lo mismo que anteriormente, solo que en lugar de elevar al cuadrado cada termino, sumamos
% al vector de gradientes el correspondiente vector de theta (de su mismo tamaño) con el primer termino anulado (igual que antes)
% y con su correspondiente termino lambda/m mutiplicado para obtener el termino de regularizacion
grad = (1/m)*(sumatorio'*X)' + (lambda/m)*(theta.*[zeros(1, size(theta,2)); ones(size(theta,1)-1, size(theta,2))]);






% =========================================================================

grad = grad(:);

end
