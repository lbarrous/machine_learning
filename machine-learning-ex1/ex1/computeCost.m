function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% (Theta traspuesta * X traspuesta) traspuesta nos dara el valor de cada elemento de la hipotesis, es decir, un
% vector 1xm en el cual tendremos en cada elemento el valor de la regresion lineal (sumatorio de los productos de cada parametro por cada elemento del ejemplo de entrenamiento),
% si hacemos la traspuesta de esa matriz podremos restar a cada elemento de la hipotesis su correspondiente valor y
 
sumatorio = (theta'*X')'-y;

% Elevar al cuadrado cada elemento del resultado anterior
sumatorio_al_cuadrado = sumatorio.^2;

% Finalmente sumamos cada elemento del vector mx1 que tenemos
sumatorio_total = sum(sumatorio_al_cuadrado);

J = (1/(2*m))*sumatorio_total;


% =========================================================================

end
