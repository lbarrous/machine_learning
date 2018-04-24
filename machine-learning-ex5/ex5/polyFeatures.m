function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

%Creamos un vector desde 1 al numero de exponente que queramos elevar
powers = [1:p];

%Para cada fila del conjunto de entrenamiento
for i = 1:size(X,1)
  
  %Creamos un vector de tamaño p, con el valor repetido de esa fila del conjunto de entrenamiento y con laplace_cdf
  % operacion wise-element de elevar (.^), elevamos cada elemento de ese vector creado con los valores repetidos
  % a cada numero de la variable powers, lo que resultara en un vector fila con cada elemento de esa fila del conjunto
  % de entrenamiento elevado a cada numero desde 1 a p, que luego almacenaremos en la fila correspondiente de la matriz X_poly.
  X_poly(i,:) = [ones(1,p).*X(i,:)].^powers;
  
end  


% =========================================================================

end
