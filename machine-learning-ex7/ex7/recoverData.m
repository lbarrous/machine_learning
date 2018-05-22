function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% You need to return the following variables correctly.
X_rec = zeros(size(Z, 1), size(U, 1));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the approximation of the data by projecting back
%               onto the original space using the top K eigenvectors in U.
%
%               For the i-th example Z(i,:), the (approximate)
%               recovered data for dimension j is given as follows:
%                    v = Z(i, :)';
%                    recovered_j = v' * U(j, 1:K)';
%
%               Notice that U(j, 1:K) is a row vector.
%               

%Para recuperar los ejemplos de entrenamiento proyectados con respecto eigenvectors correspondientes a la reduccion realizada anteriormente,
% debemos hacer el sumatorio del producto de cada ejemplo de entrenamiento proyectado por la matriz U reducida hasta el
% vector correspondiente a la reduccion que hayamos hecho (K->U(:, 1:K)), siendo U la matriz U original cogiendo
% hasta la columna correspondiente a los primeros valores hasta K.
% Para ellos simplemente hacemos la operacion vectorizada de la siguiente manera:

X_rec = Z*U(:,1:K)';

% =============================================================

end
