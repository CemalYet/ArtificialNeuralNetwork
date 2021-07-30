clear
clc
close all

load('Data_Problem1_regression.mat');
%New target vector calculation
Tnew = (9*T1+7*T2+7*T3+6*T4+3*T5)/(9+7+7+6+3);
% disp(size(Tnew));

%new dataset matrix 
dataset = [X1,X2,Tnew];

%choosen random 1000 samples each
[m,n] = size(dataset) ;
idx = randperm(m)  ;
NewDataset = dataset(idx(1:1000),:) ; 

disp(size(NewDataset));

%% 

Random=transpose(NewDataset);

[trainInd,valInd,testInd] = dividerand(1000,0.8,0.1,0.1);

Xtn=NewDataset(trainInd(1:800),:);

%disp((trainInd));

X1new = NewDataset(:,1);
X2new = NewDataset(:,2);
T = NewDataset(:,3);

X1train=Xtn(:,1);
X2train = Xtn(:,2);
Tnewtrain = Xtn(:,3);

X1train=transpose(X1train);
X2train=transpose(X2train);


%General dataset plot
xlin=linspace(min(X1),max(X1),33);
ylin=linspace(min(X2),max(X2),33);
[A,B]=meshgrid(xlin,ylin);
D=griddata(X1new,X2new,T,A,B,'cubic');
plot3(X1,X2,Tnew,'m.','MarkerSize',12)
hold on
mesh(A,B,D)
title('Dataset surface')
legend('Sample Points','Interpolated Surface','Location','NorthWest')

%% 
%Train data plot
xlin=linspace(min(X1train),max(X1train),33);
ylin=linspace(min(X2train),max(X2train),33);
[X,Y]=meshgrid(xlin,ylin);

Z=griddata(X1train,X2train,Tnewtrain,X,Y,'cubic');

plot3(X1train,X2train,Tnewtrain,'m.','MarkerSize',12)
hold on
mesh(X,Y,Z)
title('Training data surface')
legend('Sample Points','Interpolated Surface','Location','NorthWest')

%%
%%Configuration:
X=[X1new,X2new];
x=transpose(X);
disp(size(x));
t=transpose(T);
disp(size(t));
y=t;
alg1 = 'trainlm';% First training algorithm to use
alg2 = 'trainbfg';% Second training algorithm to use
alg3 = 'traincgf';% Polak-Ribiere conjugate gradient algorithm
H1 =10;
H2=10;% Number of neurons in the hidden layer
H3=10;
H4=10;
delta_epochs = [1,14,1000];% Number of epochs to train in each step
epochs = cumsum(delta_epochs);

%ReLU function in the hidden layer gives better result for algo-2 and algo-3 but for the first algo this is not the case the result is pretty bad 
%creation of networks
net1=feedforwardnet([H1 H2 H3 H4],alg1);
net2=feedforwardnet([H1 H2 H3 H4],alg2);
net3=feedforwardnet([H1 H2 H3 H4],alg3);

net1.layers{1}.transferFcn='tansig';%poslin oldugunda daha hizli iniyor ama performans daha kotu
net2.layers{1}.transferFcn='tansig';
net3.layers{1}.transferFcn='tansig';

net1.layers{2}.transferFcn='tansig';
net2.layers{2}.transferFcn='tansig';
net3.layers{2}.transferFcn='tansig';


net1=configure(net1,x,t);% Set the input and output sizes of the net
net2=configure(net2,x,t);
net3 =configure(net3,x,t);

net1.divideFcn = 'divideind';
net1.divideParam.trainInd = trainInd;
net1.divideParam.valInd   = valInd;
net1.divideParam.testInd  = testInd;

% disp(net1.divideParam.trainInd)

net2.divideFcn = 'divideind';
net2.divideParam.trainInd = trainInd;
net2.divideParam.valInd   = valInd;
net2.divideParam.testInd  = testInd;

net3.divideFcn = 'divideind';
net3.divideParam.trainInd = trainInd;
net3.divideParam.valInd   = valInd;
net3.divideParam.testInd  = testInd;

%  net1.trainParam.max_fail = 8;
%  net2.trainParam.max_fail = 8;
%  net3.trainParam.max_fail = 8;

net1.trainParam.mu = 1;
net1.trainParam.mu_dec = 0.8;
net1.trainParam.mu_inc = 1.5;
 
net1=init(net1);% Initialize the weights (randomly)
net2.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net3.iw{1,1}=net1.iw{1,1};

net2.lw{2,1}=net1.lw{2,1};
net3.lw{2,1}=net1.lw{2,1};

net2.b{1}=net1.b{1};
net2.b{2}=net1.b{2};
net3.b{1}=net1.b{1};
net3.b{2}=net1.b{2};
%% 

%trainlm algo training and simulation
net1.trainParam.epochs=delta_epochs(1);
net1=train(net1,x,t);
a11=sim(net1,x);

%epoch 2
net1.trainParam.epochs=delta_epochs(2);
net1=train(net1,x,t);
a12=sim(net1,x);

%epoch 3
net1.trainParam.epochs=delta_epochs(3);
net1=train(net1,x,t);
a13=sim(net1,x);
%% 

% algo 2 --epoch 1
net2.trainParam.epochs=delta_epochs(1);
net2=train(net2,x,t);
a21=sim(net2,x);  % simulate the networks with the input vector x

%algo 2 --epoch 2
net2.trainParam.epochs=delta_epochs(2);
net2=train(net2,x,t);
a22=sim(net2,x);

%algo 2 --epoch 3
net2.trainParam.epochs=delta_epochs(3);
net2=train(net2,x,t);
a23=sim(net2,x);
%% 
%trainlm algo training and simulation
net3.trainParam.epochs=delta_epochs(1);
net3=train(net3,x,t);
a31=sim(net3,x);

%epoch 2
net3.trainParam.epochs=delta_epochs(2);
net3=train(net3,x,t);
a32=sim(net3,x);

%epoch 3
net3.trainParam.epochs=delta_epochs(3);
net3=train(net3,x,t);
a33=sim(net3,x);

%% 
%plots
figure
subplot(3,6,1);
plot3(X1new,X2new,T,'bx',X1new,X2new,transpose(a11),'gx');
title([num2str(epochs(1)),' epochs']);
legend('target',alg1,'Location','north');

subplot(3,6,2);
plot3(X1new,X2new,T,'bx',X1new,X2new,transpose(a21),'rx');
legend('target',alg2,'Location','north');

subplot(3,6,3);
plot3(X1new,X2new,T,'bx',X1new,X2new,transpose(a31),'mx');
legend('target',alg3,'Location','north');
subplot(3,6,4);

postregm(a11,y); % perform a linear regression analysis and plot the result
subplot(3,6,5);
postregm(a21,y);
subplot(3,6,6);
postregm(a31,y);
%
subplot(3,6,7);
plot3(X1new,X2new,T,'bx',X1new,X2new,transpose(a12),'gx');
title([num2str(epochs(2)),' epoch']);
legend('target',alg1,alg2,'Location','north');
subplot(3,6,8);
plot3(X1new,X2new,T,'bx',X1new,X2new,transpose(a22),'rx');
legend('target',alg2,'Location','north');
subplot(3,6,9);
plot3(X1new,X2new,T,'bx',X1new,X2new,transpose(a32),'mx');
legend('target',alg3,'Location','north');
subplot(3,6,10);
postregm(a12,y);
subplot(3,6,11);
postregm(a22,y);
subplot(3,6,12);
postregm(a32,y);
%
subplot(3,6,13);
plot3(X1new,X2new,T,'bx',X1new,X2new,transpose(a13),'gx');
title([num2str(epochs(3)),' epoch']);
legend('target',alg1,alg2,'Location','north');
subplot(3,6,14);
plot3(X1new,X2new,T,'bx',X1new,X2new,transpose(a23),'rx');
legend('target',alg2,'Location','north');
subplot(3,6,15);
plot3(X1new,X2new,T,'bx',X1new,X2new,transpose(a33),'mx');
legend('target',alg3,'Location','north');
subplot(3,6,16);
postregm(a13,y);
subplot(3,6,17);
postregm(a23,y);
subplot(3,6,18);
postregm(a33,y);














