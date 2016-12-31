function KLval = KL_div_NNGaussianMetric_rv(data1, data2)
% Src Code: 1305090001, 1401240001
% KLval = KL_div_NNGaussianMetric_rv(randn(5,100), 10*randn(5,100) + 2)


[dim, datanum1] = size(data1);
datanum2 = size(data2, 2);

estM1 = mean(data1,2);
estM2 = mean(data2,2);
estSig1 = data1*data1'/datanum1 - estM1*estM1';
estSig2 = data2*data2'/datanum2 - estM2*estM2';


% Gaussian exact bias GML
KLest3 = 0;
invSig1 = inv(estSig1);
invSig2 = inv(estSig2);
logp1 = getLogGaussian(data1, estM1, estSig1);
logp2 = getLogGaussian(data1, estM2, estSig2);
for idata = 1:datanum1
% for idata = 14
    curdatum = data1(:,idata);
    HessianPOverP1 = invSig1*(curdatum - estM1)*(curdatum - estM1)'*invSig1 - invSig1;
    HessianPOverP2 = invSig2*(curdatum - estM2)*(curdatum - estM2)'*invSig2 - invSig2;

%     const1 = 1/exp(2/dim*logp1(idata) );
%     const2 = 1/exp(2/dim*logp2(idata) );
    const1 = 1/exp(2/dim*(logp1(idata) + log(datanum1 - 1)));
    const2 = 1/exp(2/dim*(logp2(idata) + log(datanum2)));
%     if isinf(const2)
    if exp(logp2(idata)) < 10^-30
        Met = -HessianPOverP2;
    else
        Met = (const1*HessianPOverP1 - const2*HessianPOverP2);
    end
    %             Met = (const2*HessianPOverP1 - const1*HessianPOverP2);
    %     Met = (Met' + Met)/2;
%     idata
    [V,D] = eig(Met);
    
    Evals = diag(D)';
    [Evals, sortedIndex] = sort(Evals);
    V = V(:,sortedIndex);
    PosEvalIndex = find(Evals > 0);
    NegEvalIndex = find(Evals < 0);
    
    %             regR = dim/2*Evals(end);
    regR = 0;
    
    PosEvalNum = size(PosEvalIndex, 2);
    NegEvalNum = size(NegEvalIndex, 2);
    
%     L = [V(:,PosEvalIndex)*diag(sqrt(Evals(PosEvalIndex)*sqrt(PosEvalNum) + regR)) ...
%         V(:,NegEvalIndex)*diag(sqrt(Evals(NegEvalIndex)*sqrt(NegEvalNum)*(-1) + regR))];
    L = [V(:,PosEvalIndex)*diag(sqrt(Evals(PosEvalIndex)*PosEvalNum + regR)) ...
        V(:,NegEvalIndex)*diag(sqrt(Evals(NegEvalIndex)*NegEvalNum*(-1) + regR))];
    L = L/max(max(L,[],1),[],2);
    
    p1minIndex = knnsearch((L'*data1(:,idata))', (L'*data1(:,[1:idata-1 idata+1:end]))')';
    if (p1minIndex >= idata)
        p1minIndex = p1minIndex + 1;
    end
    p2minIndex = knnsearch((L'*data1(:,idata))', (L'*data2)')'; % test, train

    d1 = norm(L'*data1(:,idata) - L'*data1(:,p1minIndex));
    d2 = norm(L'*data1(:,idata) - L'*data2(:,p2minIndex));
    KLest3 = KLest3 + (dim*log(d2/d1) + log(datanum2/(datanum1 - 1)))/datanum1;
end

KLval = KLest3;
