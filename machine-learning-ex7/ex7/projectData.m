function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

%Para proyectar los ejemplos de entrenamiento con los eigenvectors correspondientes a la reduccion que vayamos a hacer,
% debemos hacer el sumatorio del producto de cada ejemplo de entrenamiento multiplicado por la matriz U reducida hasta el
% vector correspondiente a la reduccion que vayamos a hacer (K->U(:, 1:K)), siendo U la matriz U original cogiendo
% hasta la columna correspondiente a los primeros valores hasta K.
% Para ellos simplemente hacemos la operacion vectorizada de la siguiente manera:
Z = X*U(:, 1:K);

% =============================================================

end
