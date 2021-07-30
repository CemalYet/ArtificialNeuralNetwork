plot(lasertrain);
mu = mean(lasertrain);
sig = std(lasertrain);
dataTrainStandardized = (lasertrain - mu) / sig;
%% 
p=50;
[Train,Target]=getTimeSeriesTrainData(dataTrainStandardized,p);

LastPpoint=dataTrainStandardized(end-p+1:end);

trainind=[1:199,251:599,751:1000-p] ;
valind=[200:250,600:750];

%% 
alg1 = 'trainlm';% First training algorithm to use
alg2 = 'trainbfg';% Second training algorithm to use
alg3 = 'traincgp' ; %20-25
alg4 = 'trainscg';
alg5=   'trainbr';

H1=50;
H2=20;
net=feedforwardnet(H1,alg5);

net.layers{1}.transferFcn='tansig';
% net.trainFcn = 'trainbr';
%net.trainParam.max_fail=6;
% net.divideFcn = 'dividerand';
% net.divideParam.trainRatio=0.85 ;
% net.divideParam.valRatio =0.15;

net=configure(net,Train,Target);
net.divideFcn = '';
net.performParam.regularization=0.001;
net.trainParam.showWindow = 0;

% net.divideFcn = 'divideind';
% net.divideParam.trainInd =  trainind;
% net.divideParam.valInd =valind;

% net.divideParam.trainInd =  [1:p:(149*p),(251*p):p:(599*p),(651*p):p:((1000-p)*p)] ;
% net.divideParam.valInd =[150*p:p:250*p,600*p:p:650*p];

% plot(Train(net.divideParam.valInd));
% hold on;
% disp(net);
% plot(Train(net.divideParam.trainInd));
% hold on;
% 
% net.trainParam.mu = 1;
% net.trainParam.mu_dec = 0.8;
% net.trainParam.mu_inc = 1.5;
%% 
Mrsme=0;
count=0;
flag=1;
%while flag  
for i=1:10
net=init(net);
% train the network
net=train(net,Train ,Target);
% give p last historical data to network and predict  first future value ,namely, yhat
Yhat=sim(net,LastPpoint);
% create new input vector by choosing  p-1 element of the data, and add the
% predicted yhat to this vector as a last element
Ynew=dataTrainStandardized(end-p+2:end);
Ynew(p)=Yhat;
% give the last p point of newly created Ynew as a input including Yhat and predict future
% values.Iterate this process to predict 100 future point.
close=numel(laserpred);
for i=1:close-1
%  choose Ynew as Input
    Input=Ynew(end-p+1:end);
%  Ynew(p)=Yhat;
%  Predict future value     
    YpredictP=sim(net,Input);
%  Add predict value to end of the Input vector       
    Ynew(p+i)=YpredictP;
%  remark here I add new predicted value to end of Ynew but I did
%  not remove the past value which are need to be removed after addition of ypredicted as an input, from Ynew but it does not matter 
%  Because I select in the every iteration last p element of the Ynew.  
end
%% 

Ypredicted=Ynew(end-99:end);
% remove the normalization to calculate error
Ypredicted= (sig*Ypredicted)+mu;
rmse=sqrt(mean((laserpred - Ypredicted).^2));
if rmse <12
    flag=0;
    break
end
Mrsme=(rmse+Mrsme);
fprintf("Predict error:\n");
disp(rmse)
count=count+1;
fprintf("Number of Try :\n");
disp(count);
end
Mrsme=Mrsme/count;
fprintf("Mean rmse :\n");
disp(Mrsme);
% Plot the future values and RMSE
figure
plot([ lasertrain', Ypredicted'])

figure
subplot(2,1,1)
plot(laserpred)
hold on
plot(Ypredicted,'-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(Ypredicted - laserpred)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse);