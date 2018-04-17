function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

%El primer vector de la primera capa corresponde simplemente al numero de caracteristicas de elementos de entrada por sus correspondientes caracteristicas
% (Conjunto de entrenamiento).
a1 = X;

%Computamos el calculo de la primera capa a la segunda, para hacer el calculo utilizamos la primera matriz de pesos Theta1, la cual nos dara
% una matriz de 25*5000 con los valores de la segunda capa para cada ejemplo de entrenamiento.
a2 = sigmoid(Theta1*a1'); % 25x401*401*5000 = 25x5000
% Añadimos la fila correspondiente a la neurona 'bias' --> +1
a2=[ones(1,size(a2,2)); a2];

%Finalmente computamos el calculo de la segunda capa a la tercera, para hacer el calculo utilizamos la seguna matriz de pesos Theta2, la cual nos dara
% una matriz de 10*5000 con los valores de la tercera capa para cada ejemplo de entrenamiento.
a3 = sigmoid(Theta2*a2); % 10x26 = 26*5000 = 10x5000

%Finalmente buscamos en cada fila (a3') de cada ejemplo de entrenamiento las predicciones de cada etiqueta y cogemos la mayor y su correspondiente indice.
[max_values,indices] = max(a3', [], 2);

p = indices;









% =========================================================================


end
