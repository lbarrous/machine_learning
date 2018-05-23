function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

%Para sacar la media de manera vectorizada multiplicamos la matriz de ejemplos de entrenamiento traspuesta (2x307) por un
% vector del mismo tamaño de la primera dimension de X (307x1) lleno de unos, de manera que al hacer la multiplicacion
% entre X' y este vector, obtendremos el sumatorio de cada elemento multiplicado por 1, es decir, el sumatorio total
% de cada valor de la caracteristica. Luego dividimos entre el numero total y tenemos un vector de (nx1) con la media
% de cada caracteristica.
mu = (1/m)*(X'*ones(m,1));

%Para sacar la sigma2 de manera vectorizada restamos a la matriz de ejemplos de entrenamiento original (307x2) una
% matriz hecha con repmat de la siguiente manera:
%Cogemos el vector de medias traspuesto, y lo replicamos en la primera dimension (Fila) tantas veces como
% ejemplos de entrenamiento haya, y una sola vez en la segunda dimension (Columna), de tal modo que al final,
% replicaremos dicho vector de 1x2 (mu'), tantas veces como ejemplos tengamos, por lo cual, conseguiremos restar
% a cada ejemplo la media correspondiente de cada caracteristica. Lo siguiente es elevar al cuadrado cada elemento,
% ya que para calcular su correspondiente sigma2 es necesario elevar al cuadrado la diferencia.
%Finalmente hacemos como anteriormente con la media, y multiplicamos esta matriz resultante de las restas al cuadrado traspuesta
% por un vector del mismo tamaño que el numero de ejemplos de entrenamiento, para asi multiplicar por uno cada elemento
% y obtener el sumatorio de cada elemento en un vector de 2x1 con la sigma2 de cada caracteristica.
sigma2 = (1/m)*(((X - repmat(mu',size(X,1),1)).^2)'*(ones(m,1)));




% =============================================================


end
