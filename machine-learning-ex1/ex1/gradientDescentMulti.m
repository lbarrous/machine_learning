function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %



    %(Theta traspuesta * X traspuesta) traspuesta nos dara el valor de cada elemento de la hipotesis, es decir, un
    %vector 1xm en el cual tendremos en cada elemento el valor de la regresion lineal (sumatorio de los productos de cada parametro por cada elemento del ejemplo de entrenamiento),
    %si hacemos la traspuesta de esa matriz podremos restar a cada elemento de la hipotesis su correspondiente valor y
    %Seguidamente le hacemos la traspuesta para tener un vector fila de 1xm
    sumatory = ((theta'*X')'-y)';
    
    %Hacemos un vector auxiliar de nx1 con n=numero de caracteristicas
    auxiliar = zeros(length(theta), 1);
    
    for i = 1:length(theta)
      %Almacenamos en el parametro correspondiente el producto del sumatorio anterior por el vector
      %columna de mx1 (El cual tiene los valores de x de cada ejemplo de entrenamiento de la correspondiente caracteristica al parametro que estamos calculando,
      %Esto nos dara el valor escalar correspondiente a la operacion del sumatorio en la formula de la funcion de costos
      auxiliar(i, 1) = sumatory*X(1:m, i);
    end  

    theta = theta - alpha * (1/m) * auxiliar;





    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
