clear ; close all; clc

computeCost( [1 2; 1 3; 1 4; 1 5], [7;6;5;4], [0.1;0.2] )

computeCost( [1 2 3; 1 3 4; 1 4 5; 1 5 6], [7;6;5;4], [0.1;0.2;0.3])

[theta J_hist] = gradientDescent([1 5; 1 2; 1 4; 1 5],[1 6 4 2]',[0 0]',0.01,1000);

theta
J_hist(1)
J_hist(1000)

[theta J_hist] = gradientDescent([1 5; 1 2],[1 6]',[.5 .5]',0.1,10);
theta
J_hist

[Xn mu sigma] = featureNormalize([1 ; 2 ; 3])

% result

[Xn mu sigma] = featureNormalize(magic(3))   

% result

[Xn mu sigma] = featureNormalize([-ones(1,3); magic(3)])

% result

X = [ 2 1 3; 7 1 9; 1 8 1; 3 7 4 ];
y = [2 ; 5 ; 5 ; 6];
theta_test = [0.4 ; 0.6 ; 0.8];
computeCostMulti( X, y, theta_test )

X = [ 2 1 3; 7 1 9; 1 8 1; 3 7 4 ];
y = [2 ; 5 ; 5 ; 6];
[theta J_hist] = gradientDescentMulti(X, y, zeros(3,1), 0.01, 10);

% results
theta
J_hist

X = [ 2 1 3; 7 1 9; 1 8 1; 3 7 4 ];
y = [2 ; 5 ; 5 ; 6];
[theta J_hist] = gradientDescentMulti(X, y, [0.1 ; -0.2 ; 0.3], 0.01, 10);

% results
theta
J_hist

X = [ 2 1 3; 7 1 9; 1 8 1; 3 7 4 ];
y = [2 ; 5 ; 5 ; 6];
theta = normalEqn(X,y)