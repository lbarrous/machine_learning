function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%Vamos a hacer 64 modelos con valores desde 0.01 a 30 multiplicando por 3 cada valor de sigma y C
%Comenzamos con una matriz de 8x8 para almacenar el acierto medio de cada modelo con cada set de parametros.

%Para ello simplemente haremos dos for loop anidados, para probar cada valor de C con cada valor de sigma,
% los indices del for iran de 1 a 8 (tanto del for de C(i) como del for de sigma(j), de tal manera que en cada loop
% C=C*(3^(i-1)), empezando por C=0.01 y sigma=sigma*(3^(j-1)), empezando por sigma=0.01
% --> De tal modo que en cada bucle probaremos con un valor de C y sigma multiplicado por 3, siendo el primer valor
% C*3^0=C y sigma*3^0=sigma.
%Para cada valor de C y sigma, entrenamos la SVM con los valores X e y (training set), hacemos la prediccion de
% dicho modelo con los valores de Xval (Cross validation set) que tenemos, y sacamos el porcentaje de acierto medio
% de ese modelo, una vez obtenido, guardamos el valor en el el elemento (i,j) de la matriz 8x8 correspondiente.
%Finalmente sacamos el minimo de dicha matriz para saber cual es el set de parametros optimo, y teniendo su
% indice de fila y columna lo obtenemos facilmente, ya que C = C*(3^fila_valor_minimo) y sigma = sigma*(3^columna_valor_minimo)

valores_parametros = zeros(8,8);

C = 0.01;
sigma = 0.01;

for i = 1:8
  C_prueba = C*(3^(i-1));
   for j = 1:8
     sigma_prueba = sigma*(3^(j-1));
     model = svmTrain(X, y, C_prueba, @(x1, x2) gaussianKernel(x1, x2, sigma_prueba));
     predictions = svmPredict(model, Xval);
     valores_parametros(i,j) = mean(double(predictions ~= yval));
   endfor
endfor

[minval, row] = min(min(valores_parametros,[],2));
[minval, col] = min(min(valores_parametros,[],1));

fprintf('Optimal C: %f', C*(3^(row-1)));
fprintf('Optimal sigma: %f', sigma*(3^(col-1)));

C = C*(3^(row-1));
sigma = sigma*(3^(col-1));




% =========================================================================

end
