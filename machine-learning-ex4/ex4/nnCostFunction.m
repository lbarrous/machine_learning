function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

%Ejemplo: Si la capa de entrada tuviera 5 unidades y la capa oculta 10 --> 
% Theta1 = reshape(nn_params(1:(10*(5+1)),10,6)) --> Theta1 = reshape(nn_params(1:60,10,6))
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

%Ejemplo: Si la capa oculta tuviera 10 unidades y la de salida 4 --> 
% Theta2 = reshape(nn_params(1+(10*(5+1)):end,4,11)) --> Theta2 = reshape(nn_params(61:end,4,11))
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

%El primer vector de la primera capa corresponde simplemente al numero de caracteristicas de elementos de entrada por sus correspondientes caracteristicas
% (Conjunto de entrenamiento).
a1 = X;

%Computamos el calculo de la primera capa a la segunda, para hacer el calculo utilizamos la primera matriz de pesos Theta1 de (hidden_layer_sizexinput_layer_size+1), la cual nos dara
% una matriz de (hidden_layer_sizex1) con los valores de la segunda capa para cada ejemplo de entrenamiento.
a2 = sigmoid(Theta1*a1');
% Añadimos la fila correspondiente a la neurona 'bias' --> +1
a2=[ones(1,size(a2,2)); a2];

%Finalmente computamos el calculo de la segunda capa a la tercera, para hacer el calculo utilizamos la seguna matriz de pesos Theta2, la cual nos dara
% una matriz de (num_labelsxhidden_layer_size+1) con los valores de la tercera capa para cada ejemplo de entrenamiento --> h(theta) --> num_labels*1.
h = a3 = sigmoid(Theta2*a2)';

%Matriz necesaria para recodificar la salida de y(numero real) como un vector donde tenga 1 en el lugar que corresponda
% a la etiqueta de y y 0 en todos los demas lugares.
%Ejemplo: y_aux == 3 --> [0 0 1 0 0 0 0 0 0 0]
y_aux = [1:num_labels];

%Creamos una matriz auxiliar con todo unos menos la primera columna, que es 0, con el objetivo de poner a 0 la primera
% columna de cada matriz Theta haciendo el producto wise-element de Theta por su correspondiente matriz auxiliar
auxiliar_lambda_1 = [zeros(size(Theta1,1),1) ones(size(Theta1,1),size(Theta1,2)-1)];
auxiliar_lambda_2 = [zeros(size(Theta2,1),1) ones(size(Theta2,1),size(Theta2,2)-1)];

for i = 1:m
  
  %Formula de la funcion de costes en la cual tenemos que modificar lo que obtenemos como y en la salida, ya que
  % esta, es un numero real entre 0 y 9, y lo que queremos es obtener un array logico que corresponda al valor de y
  % correspondiente --> y_aux == 3 --> [0 0 1 0 0 0 0 0 0 0], una vez hecho esa comparacion, simplemente tenemos
  % que aplicar la formula habiendo calculado anteriormente h(theta) = a3
  J = J + 1/m*(-(y_aux == y(i,:))*log(h(i,:))' - (1-(y_aux == y(i,:)))*log(1-h(i,:))');

endfor  

%Elevamos al cuadrado cada elemento de Theta, y una vez ya con nuestras matrics Theta con la primera columna puesta a 0 para evitar regularizar la primera unidades
% de cada capa (neurona 'bias'), multiplicamos y sumamos cada elemento de esta, dandonos finalmente el valor completo
% de elevar cada termino de Theta al cuadrado menos el correspondiente al término 'bias', obteniendo asi la regularizacion
% del correspondiente Theta.
J = J + (lambda/(2*m))*(sum((auxiliar_lambda_1.*Theta1.^2)(:))+sum((auxiliar_lambda_2.*Theta2.^2)(:)));

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

%Estas matrices son necesarias para la regularizacion, ya que son de la misma dimension que Theta y
% tienen 0 en la primera columna, la que corresponde a cada unidad 'bias', y 1 en los demas lugares, de tal modo, que
% al multiplicarlas de modo wise-element(.*) conseguiremos dejar la primera columna a 0 y las demas igual, para asi
% evitar regularizar los terminos correspondientes a la unidad o neurona 'bias'
reg_aux_theta_1 = [zeros(size(Theta1,1), 1) ones(size(Theta1,1), size(Theta1, 2) - 1)];
reg_aux_theta_2 = [zeros(size(Theta2,1), 1) ones(size(Theta2,1), size(Theta2, 2) - 1)];

%Para cada ejemplo de entrenamiento
for t = 1:m
  
  %Realizamos forward propagation para calcular h(theta) = a_3
  z_1 = a_1 = X(t,:)';
  z_2 = Theta1*a_1;
  a_2 = [1; sigmoid(z_2)];
  z_3 = Theta2*a_2;
  a_3 = sigmoid(z_3);
  
  %Calculamos los terminos delta de cada capa aplicando la formula correspondiente
  delta_3 = a_3 - (y_aux == y(t,:))';
  delta_2 = Theta2'*delta_3.*[1; sigmoidGradient(z_2)];
  
  %Añadimos al gradiente de cada capa su correspondiente valor (Quitamos de delta_2 el primer termino correspondiente a la unidad 'bias'
  Theta1_grad = Theta1_grad + delta_2(2:end) * a_1';
  Theta2_grad = Theta2_grad + delta_3 * a_2';
  
endfor  


%Una vez tenemos el gradiente de ambas capas obtenido para cada ejemplo, dividimos entre el numero de ejemplos y obtenemos
% el gradiente medio de cada uno, añadimos tambien la regularizacion, obtenida de sumar al gradiente su correspondiente
% termino de Theta de su capa multiplicado por lambda y dividido por el numero de ejemplos de entrenamiento
% (La multiplicacion wise-element con la matriz auxiliar es para eliminar la regularizacion del termino 'bias')
% Y con esto obtendriamos las derivadas parciales de cada parametro que necesitamos.
Theta1_grad = Theta1_grad/m + (lambda/m)*reg_aux_theta_1.*Theta1;
Theta2_grad = Theta2_grad/m + (lambda/m)*reg_aux_theta_2.*Theta2;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
