function [U,S,V] = mat_recovery_set(X0,IndexM,r,tol)

fprintf('SET ----- Noiseless case \n');

[m,n] = size(X0);
U = orth(randn(m,r));
X0n2 = norm(X0,'fro')^2;
X0n2_tol = tol*X0n2;

itN = 50;
epsilon = 5e-3;
pn = 1;
while 1
    if mod(pn,2) == 1
        fprintf('The %d Process: Normal Update\n',pn);
        [U,S,V,Xr_norm] = NormalUpdate(X0,IndexM,U,tol,itN,epsilon);
        if min(Xr_norm) < X0n2_tol
            return;
        end
        
        fprintf('The %d Process: Block Update\n',pn);
        [U,S,V,Xr_norm] = BlockUpdate(X0,IndexM,U,tol,itN,epsilon);
        if min(Xr_norm) < X0n2_tol
            return;
        end
    else
        fprintf('The %d Process: Normal Update\n',pn);
        [V,S,U,Xr_norm] = NormalUpdate(X0',IndexM',V,tol,itN,epsilon);
        if min(Xr_norm) < X0n2_tol
            return;
        end
        
        fprintf('The %d Process: Block Update\n',pn);
        [V,S,U,Xr_norm] = BlockUpdate(X0',IndexM',V,tol,itN,epsilon);        
        if min(Xr_norm) < X0n2_tol
            return;
        end
    end
    pn = pn+1;
    if pn > 10
        break;
    end
end


function [U,S,V,Xr_norm] = NormalUpdate(X0,IndexM,U,tol,itN,epsilon)
[m,n] = size(X0);
X0n2 = norm(X0,'fro')^2;
X0n2_tol = tol*X0n2;
itn = 1;
Xr_norm = zeros(1,itN);

tic;
VS = Factorization01(X0,IndexM,U,[]);
V = orth(VS);
while 1
    [V,US,tvV,fvalV] = GMLineMoveSplit1d01(X0',IndexM',eye(n),V,8);
    U = orth(US);
    [U,VS,tv,fval] = GMLineMoveSplit1d01(X0,IndexM,eye(m),U,8);
    [V S VV] = svd(VS,0);
    U = U*VV;
    fval_min = min(fval);
    Xr_norm(itn) = fval_min;

    if fval_min < X0n2_tol
        Xr_norm = Xr_norm(1:itn);
        return;
    end
    if 1-fval_min/fval(1) < epsilon
        [UOld,UNew,h,Flag] = GM0BNDetectionUB02(X0,IndexM,eye(m),eye(m),U);
        if Flag == 1
            U = UNew;
            VS = Factorization01(X0,IndexM,U,[]);
            V = orth(VS);
        elseif 1-fval_min/fval(1) < epsilon/5
            break;
        end
    end

    if mod(itn,10) == 0
        fprintf('itn=%5d: Xr_norm=%.8f: time=%f \n',itn,fval_min/X0n2,toc);
        tic;
    end 
    if itn == itN
        break;
    end
    itn = itn+1;
end

function [U,S,V,Xr_norm] = BlockUpdate(X0,IndexM,U,tol,itN,epsilon)
X0n2 = norm(X0,'fro')^2;
X0n2_tol = tol*X0n2;
itn = 1;
Xr_norm = zeros(1,itN);

tic;
[UB,UBp] = SpaceSplit03(X0,IndexM,U);
while 1
    [Uc,VS,tv,fval] = GMLineMoveSplit1d01(X0,IndexM,UB,UB'*U,8);
    [V S VV] = svd(VS,0);
    U = (UB*Uc)*VV;
    fval_min = min(fval);
    Xr_norm(itn) = fval_min;

    if fval_min < X0n2_tol
        Xr_norm = Xr_norm(1:itn);
        return;
    end
    if 1-fval_min/fval(1) < epsilon
        [UOld,UNew,h,Flag] = GM0BNDetectionUB02(X0,IndexM,UB,UBp,UB'*U);
        if Flag == 1
            U = UB*UNew;
        elseif 1-fval_min/fval(1) < epsilon/5
            break;
        end
    end

    if mod(itn,10) == 0
        fprintf('itn=%5d: Xr_norm=%.8f: time=%f \n',itn,fval_min/X0n2,toc);
        tic;
    end      
    if itn == itN
        break;
    end

    itn = itn+1;
end
