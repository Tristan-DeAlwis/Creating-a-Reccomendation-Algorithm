function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

J = 0.5 .* sum(sum((((Theta * X')' - Y) .* (R == 1)).^2));
J = J + lambda*sum(sum(Theta.^2))/2 + lambda*sum(sum(X.^2))/2; 
            
diff = (X*Theta'-Y);

X_grad = (diff.*R)*Theta;                 %unregularized vectorized implementation
Theta_grad = ((diff.*R)'*X);              %unregularized vectorized implementation


X_grad = X_grad + (lambda * X);             % regularized
Theta_grad = Theta_grad + (lambda * Theta);  % regularized

grad = [X_grad(:); Theta_grad(:)];

end
