function rmse = projectionErr(dataSet,q)
    covarianceMatrix=cov(dataSet);
    [E,~]=eigs(covarianceMatrix,q);
    z=dataSet*E;
    F=E';
    Regenerated=z*F;
    mu=mean(dataSet,2);
    disp(size(mu))
    Regenerated=Regenerated+mu;
    rmse=sqrt(mean(mean((dataSet- Regenerated).^2)));
end