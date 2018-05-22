function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

%En primer lugar tendremos que hacer todas las operaciones para cada centroide.

%Devolvemos una matriz con tantas columnas como centroides tengamos, y en cada fila tenemos un el valor medio deal
% todos los puntos asignados al correspondiente centroide (El mas cercano a cada uno). Es decir, obtenemos el espacio
% medio en el que estan los puntos asociados a cada centroide.

%Para obtenerlo simplemente obtenemos un vector binario comparando el indice de centroide de cada ejemplo de entrenamiento
% con el valor como tal, obteniendo un vector binario teniendo 1 en cada fila de cada ejemplo que este asociado a ese
% centroide, y 0 en todos los demas. Una vez obtenido, hacemos la suma de ese vector binario para obtener cuantos
% elementos tenemos para ese centroide, y para obtener el factor de multiplicacion 1/num_elementos_centroide.

%El segundo factor es la suma de todos los ejemplos de entrenamiento correspondientes a ese centroide, para ello
% obtenemos el mismo vector binario anterior, y multiplicamos wise-element(.*) por cada elemento de X (ejemplos de entrenamiento),
% de tal manera, que asi anularemos los elementos que no correspondan a ese centroide multiplicando por el 0 correspondiente
% del vector binario (los valores de X que no correspondan al centroide actual tendran asociado un valor 0 en el vector binario
% y se anularan al multiplicarse), y dejaremos igual los demas elemento que si correspondan al centroide (los valores de X que
% si correspondan al centroide quedaran multiplicados por el 1 correspondiente del vector binario y quedaran igual).
% Finalmente, con hacer la suma por columnas del vector que obtenemos, nos resultara en un vector que suma
% todas las columnas (Que a efectos practicos es como sumar todas las filas de la matriz, como si cada una fuera un vector
% individual, con el detalle de que hemos anulado las filas de X que no pertenecian al centroide).
for i = 1:K
  centroids(i,:) = 1/(sum(idx == i))*(sum((idx == i).*X));
endfor  





% =============================================================


end

