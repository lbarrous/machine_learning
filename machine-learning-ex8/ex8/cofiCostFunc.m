function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

%Para sacar el coste de cada pelicula con cada usuario tenemos que multiplicar nuestra matriz X (Peliculas con caracteristicas)
% por nuestra matriz Theta para hacer las predicciones, y restarle el valor real del rating que le hayamos dado, y hacer
% el valor cuadratico. Tras esto, tendremos una matriz con la diferencia entre la prediccion que tenemos de cada pelicula con cada usuario
% y el valor real que le hemos dado. Debido a que habra usuarios que no han valorado alguna pelicula, tendremos valores que no tendran
% sentido, por lo cual, tenemos que multiplicarlo wise-element(.*) por la matriz R, que indica que peliculas han sido
% valoradas por que usuario, para asi solo tener los valores de las peliculas realmente valoradas. (Con esta multiplicacion aprovechamos
% el 0 de los valores no valorados para eliminar los valores sin sentido de la matriz)
%Tras esto, hacemos un sumatorio de todos los elementos de la matriz con los errores calculados entre cada prediccion
% y cada valor real para saber el error acumulado
%Finalmente añadimos el sumatorio de la regularizacion, es decir, el sumatorio de todos los elementos de Theta al cuadrado
% y el sumatorio de todos los elementos de X al cuadrado (Es decir, la regularizacion de ambos terminos que vamos a minimizar, por un lado
% X y por otro lado Theta), multiplicados por su factor lambda.
J = (1/2)*sum(sum(((((X*Theta')-Y).^2).*R))) + (lambda/2)*sum(sum((Theta).^2)) + (lambda/2)*sum(sum((X).^2));

%Para calcular los gradientes de manera vectorial procedemos a realizar las multplicaciones de matrices:
%Tenemos X(5x3), Theta (4x3), Y(5x4), R(5x4)
%Al multplicar X*Theta' Obtenemos una matriz de 5x4 que podemos restar con la matriz Y, para obtener los valores de predicciones,
% tras esto, procedemos como antes multiplicando wise-element(.*) por R para eliminar valores sin sentido de valoraciones no hechas por el usuario a una pelicula.
% Seguidamente multiplicamos esta matriz 5x4 por Theta(4x3), obteniendo finalmente la matriz 5x3 con los valores de los gradientes
% de cada pelicula para cada caracteristica con respecto a cada caracteristica.
X_grad = ((X * Theta' - Y) .* R) * Theta + lambda*X;
%Al multplicar X*Theta' Obtenemos una matriz de 5x4 que podemos restar con la matriz Y, para obtener los valores de predicciones,
% tras esto, procedemos como antes multiplicando wise-element(.*) por R para eliminar valores sin sentido de valoraciones no hechas por el usuario a una pelicula.
% Seguidamente multiplicamos esta matriz traspuesta (4x5) por X(5x3), obteniendo finalmente la matriz 4x3 con los valores de los gradientes
% de cada usuario para cada caracteristica con respecto a cada caracteristica.
Theta_grad = ((X * Theta' - Y) .* R)' * X + lambda * Theta;  




% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
