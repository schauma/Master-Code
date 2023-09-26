close all
clear variables
rng('shuffle')

Nneuron = 8;
dt  = 0.1e-3;
TE = 2; 
t = 0:dt:TE;
M = int64(TE/dt) + 1;

lambdaV = 0;
sigmaV = 0*1e-9;
mu = 0*1e-2;
nu = 0*1e-1;


color = jet(round(1.5*Nneuron));
neuron_split_ration = Nneuron -floor(0.5*Nneuron);

%Ratio of rDec and iDec dependent
rDec = 200*[ones(1,neuron_split_ration),-ones(1,Nneuron-neuron_split_ration)];
iDec = 1*0.5*[ones(1,neuron_split_ration),-ones(1,Nneuron-neuron_split_ration)];

% rDec = 200*[ones(2,neuron_split_ration),-ones(2,Nneuron-neuron_split_ration)];
% iDec = 0.5*[ones(2,neuron_split_ration),-ones(2,Nneuron-neuron_split_ration)];
%  
% DO NOT DELETE
iDec = randn(2,Nneuron);
iDec = 0.5*iDec./vecnorm(iDec,2,1);
rDec = 500*iDec;
%plot_thresholds(rDec,Nneuron,color)
% hold off

% Examples :
ratio = 0.04;
window = 10;
x_hat = floor(M*ratio);
f = 1000;

% Example 0
xT = 20*[1-cos(pi*t);pi*sin(pi*t)];
dxT = 20*[pi*sin(pi*t);pi^2*cos(pi*t)];

% Example 1 
xT = f*smoothdata([zeros(1,floor(M*ratio)),ones(1,M-floor(M*ratio))],"gaussian",50);
dxT = f/(window*dt).*[zeros(1,x_hat-window/2),...
                ones(1,window),...
                zeros(1,M-x_hat-window/2)];
xT = [xT;dxT];
dxT = [dxT;zeros(1,M)];
 
% % Example 2
% xT = f/10*sin(2*pi*t);
% dxT = f/10*2*pi*cos(2*pi*t);

% Example 3
xT = xT(1,:);
dxT = dxT(1,:);


% Example 4
%xT = [xT(1,:);xT(1,:)];
%dxT = zeros(2,M);

%Example 5 
xT = 200*[sin(2*pi*t);2*pi*cos(2*pi*t)];
dxT = 200*2*pi*[cos(2*pi*t);-2*pi*sin(2*pi*t)];


% Example 6
% t0 =8;
% t1 =6;
% a = 10;
% xT1 = 1./(1+exp(-(a*t-t0)));
% xT2 = 1./(1+exp(-(a*t-t1)));
% dxT1 = a*exp(-(a*t-t0)).*xT1.^2;
% dxT2 = a*exp(-(a*t-t1)).*xT2.^2;
% aexp = exp((a*t-t0));
% ddxT1 =(a.^2.*exp(a*t + t0).*(exp(t0) - exp(a*t)))./(exp(a*t) + exp(t0)).^3;
% dddxT1 = (3*a^3.*exp(t0 + a*t).*exp(a*t).*(exp(a*t) - exp(t0)))./(exp(a*t) + exp(t0)).^4 - (a^3.*exp(t0 + a*t).*(exp(a*t) - exp(t0)))./(exp(a*t) + exp(t0)).^3 - (a^3.*exp(t0 + a*t).*exp(a*t))./(exp(a*t) + exp(t0)).^3;
% ddddxT1 = -(a^4.*(exp(t0 + 4*a*t) - exp(4*t0 + a*t) - 11*exp(2*t0 + 3*a*t) + 11*exp(3*t0 + 2*a*t)))./(exp(5*t0) + 5*exp(t0 + 4*a*t) + exp(5*a*t) + 5*exp(4*t0 + a*t) + 10*exp(2*t0 + 3*a*t) + 10*exp(3*t0 + 2*a*t));
% % xT = 100*[1;0.5].*[xT1;xT2];
% % dxT = 100*[1;0.5].*[dxT1;dxT2];
% xT = 5*[xT1;dxT1;ddxT1;dddxT1];
% dxT = 5*[dxT1;ddxT1;dddxT1;ddddxT1];



lambdaD = 10;
iDec= 0.1*[1,-1,0,0;0,0,-1,1];
iDec = 0.1*[eye(4),-eye(4)];
%iDec= 0.1*[-1,1];
rDec =1*iDec;
B = eye(4);

A = [0;10].*[0,1;-5,-10];
A = 1000*[   -0.9362    0.5566    0.8155   -0.1290;
   -1.4567   -0.6816    0.2176   -0.6465;
    0.4937   -1.0893   -0.9542    0.1516;
   -1.8513    1.4284   -0.7552   -0.6146];
C = eye(4);
ALLOW_LEARNING= 1;
[xE,V,rate,spikes,error,feedforward] = controller(A,B,C,xT,dxT,Nneuron,dt,M,rDec,iDec,mu,nu,lambdaD,sigmaV,lambdaV,ALLOW_LEARNING);


[err,l1,l2,L1,L2_squared] = calculateError(xT,xE,dt);

disp(sum(spikes,"all"))

% figure
% plot_spikes(t,spikes,100,0.25,-0.5,color)


% x changing the factor       Does affect error/spikes
% y changing the base         Doesnt affect error 
% z changing my change        Does affect both (for high lambdaD)
% u changing the lambdaD      Does affect the error/spikes without change
% Results error/ # spikes

% plot x lambdaD y error
% plot x lambdaD y error with my change
% plot x factor,y base z error with my change
% just mention it changes the number of spikes 
%f = figure();
%f.Position =[0,600,1400,500];
figure
plot (t,xE,"LineWidth",2)
grid on
hold on
plot(t,xT,"LineWidth",2,"Color","k");
hold off
figure
plot(t,xT-xE)


