clear
clc
close all

load('threes','-ascii');

threesStandart=mapstd(threes);

% 1.Compute the mean 3 and disp mean 3
Mean3=mean(threes,2);
disp((Mean3))
%threes=threes-Mean3;
imagesc(reshape(threes(50,:),16,16),[0,1]);
% 2.compute covariance of whole 3's and plot it
covThrees=cov(threes);
[Vthrees,Dthrees] = eig(covThrees);
%plot(diag(Dthrees))
%% 

% 3.Compress the dataset by projecting it onto one, two, three, and four principal components. Now reconstruct the images
% from these compressions and plot some pictures of the four reconstructions.
[E1,D1]=eigs(covThrees,1);
[E2,D2]=eigs(covThrees,2);
[E3,D3]=eigs(covThrees,3);
[E4,D4]=eigs(covThrees,4);

z1=threes*E1;
z2=threes*E2;
z3=threes*E3;
z4=threes*E4;

R1=z1*E1';
R2=z2*E2';
R3=z3*E3';
R4=z4*E4';


[U1,Z1]=pca(threesStandart,'NumComponents',1 );
[U2,Z2]=pca(threesStandart,'NumComponents',2 );
[U3,Z3]=pca(threesStandart,'NumComponents',3 );
[U4,Z4]=pca(threesStandart,'NumComponents',4 );

r1=Z1*U1';
r2=Z2*U2';
r3=Z3*U3';
r4=Z4*U4';


figure
subplot(2,2,1)
imagesc(reshape(R1(1,:),16,16),[0,1]);
subplot(2,2,2)
imagesc(reshape(R2(1,:),16,16),[0,1]);
subplot(2,2,3);
imagesc(reshape(R3(1,:),16,16),[0,1]);
subplot(2,2,4);
imagesc(reshape(R4(50,:),16,16),[0,1]);
%% 
% Write a function which compresses the entire dataset by projecting it onto q principal components, then reconstructs it and
% measures the reconstruction error. Note that by choosing how many eigenvectors we use to reconstruct the image we are
% fixing the number of components, and the quality of the reconstruction. Now call this function for values of q from 1 to 50
% (here you probably want to use a loop) and plot the reconstruction error as a function of q.

for q=1:50
    [rmse]=projectionErr(threes,q);
    rmseA(q)=rmse;
end

figure
subplot(2,1,1)
plot(rmseA);
hold on
subplot(2,1,2)
plot(diag(Dthrees));
%% 
% What should the reconstruction error be if q = 256? What is it if you actually try it? Why?
%because most of the eigen values are zero so increasing diensionality does not help after 50 point.because only 50 values are not zero
[rmse]=projectionErr(threes,256);
disp(rmse);
%% 
% Use the Matlab function cumsum to create a vector whose i-th element is the sum of all but the i largest eigenvalues for
% i = 1 : 256. Compare the first 50 elements of this vector to the vector of reconstruction errors calculated previously. What
% do you notice?

[Vcumsum,Dcumsum]=eigs(covThrees,256);
sumE=sum(diag(Dcumsum));
LargestV=cumsum(diag(Dcumsum));
normalize=(LargestV/sumE)*100;
plot(normalize);
disp(rmseA);
disp(LargestV(1:50));

