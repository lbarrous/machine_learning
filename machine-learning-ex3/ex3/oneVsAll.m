function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

%Inicializamos theta con todo zeros como valor auxiliar para cada comenzar a ejecutar la funcion y sacar el valor theta
% optimo. (Comenzamos con theta todo ceros)
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);

%Vamos a generar una fila de la matriz por cada clasificador (Es decir, por cada numero distinto que pueda ser)
for c = 1:num_labels
  
  %Obtenemos con fmincg los valores de theta optimos para el clasificador correspondiente a ese numero
  % (Por lo cual, obtenemos un vector theta con los valores optimos para la clasificacion de esa etiqueta correspondiente,
  % en este caso 'c', que corresponde al numero que queremos clasificar y comprobar si cada ejemplo corresponde
  % a esa etiqueta).
  %Ejecutamos la funcion para cada uno, y como vector 'y', le pasamos un vector resultante de hacer la comparacion
  % (y == c), esto nos da un vector con unos en cada ejemplo que satisfaga que la salida corresponde a esea etiqueta, y
  % ceros, en donde no corresponda. De este modo estamos realizando distintos clasificadores, en los cuales, cada
  % vector 'y' de salida que le pasamos correspondera a un vector con unos en los ejemplos que corresponda a la etiqueta
  % que estamos clasificando, de modo que el clasificador en curso se basara solo en los resultados del conjunto de datos X
  % que correspondan a la etiqueta que se esta clasificando.
  % EJEMPLO: cuando c = 4 --> Le pasamos un vector 'y', donde tendra unos en los ejemplos que sean 4, y ceros en todos los demas,
  % de tal modo que el clasificador, intentara buscar unos parametros theta basados en un vector de salida en el cual solo
  % toma como resultados positivos los de esa etiqueta en concreto.
  [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
  
  %Finalmente añadimos a la matriz all_theta la correspondiente fila de thetas para el clasificador de esa etiqueta
  all_theta(c, :) = theta';
  
endfor  










% =========================================================================


end
