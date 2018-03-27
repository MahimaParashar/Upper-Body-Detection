[X, Y, trRegs] = HW2_Utils.getPosAndRandomNegHelper("train");
load('trainAnno.mat');
    
[w, b, alpha] = trainSVM(X, Y);
ap_list = zeros(10,1);
h = 1;

 for iter = 1:10
    A = find((alpha < 0.0002) & (Y == -1) & (h == 1));
    X(:, A) = [];
    Y(A, :) = [];
    
    imagefiles = dir('hw2data/trainIms/*.jpg');
    nfiles = length(imagefiles);
    for ii=1:nfiles
        currentfilename = imagefiles(ii).name;
        currentimage = imread(currentfilename);
        images{ii} = currentimage;
    end
    
    for i = 1:nfiles
       im = images{i};
       rects = HW2_Utils.detect(im,w,b,0);
       ubs = ubAnno{i};
       for j=1:size(ubs,2)
            overlap = HW2_Utils.rectOverlap(rects, ubs(:,j));                    
            rects = rects(:, overlap < 0.3);
            if isempty(rects)
                break;
            end
       end
       
       rects(1:4,:) = uint8(rects(1:4,:));
        
       nNeg2SamplePerIm = 2;
       [D_i, R_i] = deal(cell(1, nNeg2SamplePerIm));
       for j=1:nNeg2SamplePerIm
           imReg = im(rects(2,j):rects(4,j), rects(1,j):rects(3,j),:);
           imReg = imresize(imReg, HW2_Utils.normImSz);
           R_i{j} = imReg;
           D_i{j} = HW2_Utils.cmpFeat(rgb2gray(imReg));                    
       end
       negD{i} = cat(2, D_i{:});                
    end
    B = cat(2, negD{:});
    negD = {};
    X = [X B];
    Y = [Y; -1*ones(size(B,2),1)];
    
    [w, b, alpha] = trainSVM(X, Y);
    Y_pred = (w'*X + b)';
    Y_pred(Y_pred > 1) = 1;
    Y_pred(Y_pred <= 1) = -1;

    acc = sum(Y_pred == Y)/size(Y_pred,1);
    HW2_Utils.genRsltFile(w, b, "val", "111463071.mat");
    [ap, prec, rec] = HW2_Utils.cmpAP("111463071.mat", "val");
    
    ap_list(iter, 1) = ap;
    iter
 end
 
HW2_Utils.genRsltFile(w, b, "test", "111495982.mat");
 
 function [w, b, alpha] = trainSVM(X, Y)
    [d, n] = size(X);
    C = 0.1;
    
    H = double((Y*Y').*(X'*X) + eye(n)/C);
    f = -1*ones(1,n);
    Aeq = Y';
    Beq = 0;
    A = zeros(1,n);
    b = 0;

    lb = zeros(n, 1);
    ub = Inf * ones(n,1);

    alpha = quadprog(H, f, A, b, Aeq,Beq,lb, ub);
    w = X * (alpha .* Y);
    i = min(find((alpha >= 0) & (Y == 1)));
    T = X'*X;
    b = 1 - T(i,:)*(alpha.*Y);
end 
