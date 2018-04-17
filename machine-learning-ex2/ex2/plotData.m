function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%




% Find devuelve un vector columna con todos los indices del vector en el que buscamos que tiene valores que cumplan la expresion
% Ejemplo: A = [1; 3; 0; 0; 1; 0] --> find(A>0) == [1; 2; 5];
pos = find(y==1); neg = find(y == 0);
% Dibuja los valores de X que corresponden a y = 1 (Cada X(pos, 1) y X(pos,2) es un ejemplo positivo)
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
% Dibuja los valores de X que corresponden a y = 0 (Cada X(neg, 1) y X(neg,2) es un ejemplo negativo)
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);




% =========================================================================



hold off;

end
