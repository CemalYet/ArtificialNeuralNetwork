clear
clc
close all

%%%%%%%%%%%
%algorlm.m
% A script comparing performance of 'trainlm' and 'trainbfg'
% trainbfg - BFGS (quasi Newton)
% trainlm - Levenberg - Marquardt
%%%%%%%%%%%

% Configuration:
alg1 = 'trainlm';% First training algorithm to use
alg2 = 'trainbfg';% Second training algorithm to use
alg3 = 'trainbr';
H = 350;% Number of neurons in the hidden layer
delta_epochs = 1000;% Number of epochs to train in each step
epochs = cumsum(delta_epochs);

%generation of examples and targets
dx=0.01;% Decrease this value to increase the number of data points
x=0:dx:3*pi;
y=sin(x.^2);
sigma=0.2;% Standard deviation of added noise
yn=y+sigma*randn(size(y));% Add gaussian noise
t=y;% Targets. Change to yn to train on noisy data

disp(size(x))
disp(size(t))
%% 

[trainInd,valInd,testInd] = dividerand(943,0.8,0,0.2);
Xn=x(trainInd(1:750));
Xt=x(testInd(1:189));
Tn=t(trainInd(1:750));
Tt=t(testInd(1:189));

%% 

%creation of networks
net1=feedforwardnet(H,alg1);% Define the feedfoward net (hidden layers)
net2=feedforwardnet(H,alg2);
net3=feedforwardnet(H,alg3);

net1=configure(net1,Xn,t);% Set the input and output sizes of the net
net2=configure(net2,x,t);
net3=configure(net3,x,t);

net1.divideFcn = 'dividetrain';% Use training set only (no validation and test split)
net2.divideFcn = 'dividetrain';


net1=init(net1);% Initialize the weights (randomly)



net2.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net2.lw{2,1}=net1.lw{2,1};
net2.b{1}=net1.b{1};
net2.b{2}=net1.b{2};
net3.lw{2,1}=net1.lw{2,1};
net3.b{2}=net1.b{2};

%training and simulation
net1.trainParam.epochs=delta_epochs(1);  % set the number of epochs for the training 
net2.trainParam.epochs=delta_epochs(1);
% net3.trainParam.epochs=delta_epochs(1);
%% 

net1=train(net1,Xn,Tn);   % levenberg
y=net1(Xt);
perf=perform(net1,y,Tt);
disp(perf);
%% 

net2=train(net2,Xn,Tn);%newton
y=net2(Xt);
perf=perform(net2,Tt,y);
disp(perf);
%% 

net3=train(net3,x,t);%bayes

%% 

a11=sim(net1,x); a21=sim(net2,x); a31=sim(net3,x); % simulate the networks with the input vector x

% net1.trainParam.epochs=delta_epochs(2);
% net2.trainParam.epochs=delta_epochs(2);
% net3.trainParam.epochs=delta_epochs(2);
% net1=train(net1,x,t);
% net2=train(net2,x,t);
% net3=train(net3,x,t);
% a12=sim(net1,x); a22=sim(net2,x); a32=sim(net3,x);
% 
% net1.trainParam.epochs=delta_epochs(3);
% net2.trainParam.epochs=delta_epochs(3);
% net3.trainParam.epochs=delta_epochs(3);
% net1=train(net1,x,t);
% net2=train(net2,x,t);
% net3=train(net3,x,t);
% a13=sim(net1,x); a23=sim(net2,x);a33=sim(net3,x);

% %plots
% figure
% subplot(4,4,1);
% plot(x,t,'bx',x,a11,'r',x,a21,'g'); % plot the sine function and the output of the networks
% title([num2str(epochs(1)),' epochs']);
% legend('target',alg1,alg2,'Location','north');
% subplot(4,4,2);
% postregm(a11,y); % perform a linear regression analysis and plot the result
% subplot(4,4,3);
% postregm(a21,y);
%
% subplot(3,3,4);
% plot(x,t,'bx',x,a12,'r',x,a22,'g');
% title([num2str(epochs(2)),' epoch']);
% legend('target',alg1,alg2,'Location','north');
% subplot(3,3,5);
% postregm(a12,y);
% subplot(3,3,6);
% postregm(a22,y);
% %
% subplot(3,3,7);
% plot(x,t,'bx',x,a13,'r',x,a23,'g');
% title([num2str(epochs(3)),' epoch']);
% legend('target',alg1,alg2,'Location','north');
% subplot(3,3,8);
% postregm(a13,y);
% subplot(3,3,9);
% postregm(a23,y);
