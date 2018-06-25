function [ Xopt ] = mat_recovery_nclrsdp( X, W )

fprintf('NclrSdp ----- Noiseless case \n');
[M,N] = size(X);
cvx_begin sdp quiet
        variable A1(M,M) symmetric;
        variable A2(N,N) symmetric;
        variable  Xopt(M,N);
        minimize 0.5*(trace(A1)+trace(A2));
        subject to
            [A1 Xopt; Xopt.' A2] >= 0;
            X .* W == Xopt .* W;
cvx_end

end
