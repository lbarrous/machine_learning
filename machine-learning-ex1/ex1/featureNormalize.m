function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

% mean(A) coge una matriz y le hace la media a cada columna, dando un vector 1xsize(A,2) --> vector fila con media de cada columna
mu = mean(X);
% std(A) coge una matriz y le hace la desviacion estandar a cada columna, dando un vector 1xsize(A,2) --> vector fila con desviacion estandar de cada columna
sigma = std(X);

%repmat(A,m,n) coge una matriz o vector y lo replica m veces en la dimension vertical y n veces en la dimension horizontal
% ejemplo: repmat([1 2; 3 4], 2, 3) = (1 2 1 2 1 2; 3 4 3 4 3 4; 1 2 1 2 1 2; 3 4 3 4 3 4;)

%Hacemos una matriz del mismo tamaño de X, en la cual, cada columna es el valor de la media de cada columna de X, repetido
% en vertical, para poder restarselo a la matriz X.

%Ejemplo: X=[1 2; 3 4; 5 6]; --> X_mu = [4.5 6; 4.5 6; 4.5 6]
X_mu = repmat(mu, size(X,1), 1);
%Con sigma hacemos lo mismo
X_sigma = repmat(sigma, size(X,1), 1);

%Restamos a cada valor su correspondiente media
X_norm = X_norm - X_mu;
%Dividimos cada valor por su correspondiente desviacion (Para el operador ./ es necesario que las matrices sean de igual dimension)
X_norm = X_norm ./ X_sigma;







% ============================================================

end
