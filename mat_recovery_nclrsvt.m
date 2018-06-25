function [ Xopt ] = mat_recovery_nclrsvt( X, W, delta, tau, maxIter )

fprintf('NclrSvt ----- Noiseless case \n');
[M,N] = size(X);
Y = zeros(M,N);

for k = 1:maxIter
    [U,S,V] = svd(Y);
    S = S-tau;
    S = S.*double(S>0);
    
    Xopt = U*S*V';
    Y = Y+delta*(X-Xopt).*W;
end


end
