[X, Y, X_val, Y_val, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg();

[d, n] = size(X);

C = 0.1;
H = (Y*Y').*(X'*X);
f = -1*ones(1,n);
Aeq = Y';
Beq = 0;
A = zeros(1,n);
b = 0;

lb = zeros(n, 1);
ub = C * ones(n,1);

alpha = quadprog(H, f, A, b, Aeq,Beq,lb, ub);
w = X * (alpha .* Y);
i = min(find((alpha >= 0) & (Y == 1)));
T = X'*X;
b = 1 - T(i,:)*(alpha.*Y);

HW2_Utils.genRsltFile(w, b, "val", "outputFile");