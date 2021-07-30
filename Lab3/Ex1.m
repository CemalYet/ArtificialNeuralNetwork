dataSet=(randn(50,500));
dataSet=dataSet';
mu=mean(dataSet);
disp(size(mu));

%% 
covarianceMatrix=cov(dataSet);
%disp(size(covarianceMatrix));
[Vrand1,Drand1] = eig(covarianceMatrix);
[emax,emax_ind]=max(diag(Drand1));
u=Vrand1(:,emax_ind);
%sumofp=sum(diag(Vrand1));
disp(u);
[Vrand,Drand]=eigs(covarianceMatrix,50);

sumoflarg=sum(diag(Drand));
%disp((Drand));
%disp(sumofp-sumoflarg);
z=dataSet*Vrand;
F=Vrand';
A=z*F;
A=A+mu;
rmse=sqrt(mean(mean((dataSet-A).^2)));
disp(rmse);