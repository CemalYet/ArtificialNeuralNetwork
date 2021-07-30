mu = mean(lasertrain);
sig = std(lasertrain);
dataTrainStandardized = (lasertrain - mu) / sig;
p=10;
LastPpoint=dataTrainStandardized(end-p+1:end);
[Train,Target]=getTimeSeriesTrainData(dataTrainStandardized ,p);

disp(Target);
numFeatures = p;
numResponses = 1;
numHiddenUnits =100;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',650, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

%% 
net = trainNetwork(Train,Target,layers,options);
%% 
%disp(Target(end-p+1:end));
net = predictAndUpdateState(net,Train);
[net,Yhat] = predictAndUpdateState(net,LastPpoint);
Ynew=dataTrainStandardized(end-p+2:end);
Ynew(p)=Yhat;
close=numel(laserpred);
for i = 1:close-1
    Input=Ynew(end-p+1:end);
    [net,YpredictP] = predictAndUpdateState(net,Input,'ExecutionEnvironment','cpu');
     Ynew(p+i)=YpredictP;
end
YPred=Ynew(end-99:end);
YPred = sig*YPred + mu;
rmse = sqrt(mean((YPred-laserpred).^2));
disp(mean(rmse));

%% 
figure
plot([ lasertrain', YPred'])

figure
subplot(2,1,1)
plot(laserpred)
hold on
plot(YPred,'-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(YPred - laserpred)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse);





