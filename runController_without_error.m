% close all 
clf
clear variables
rng('shuffle')

Nneuron = 100;
dt  = 0.1e-2;
TE = 3; 
t = dt:dt:TE;
M = int64(TE/dt);

lambdaV = 0;
sigmaV = 1;
mu = 0.0001;
nu = 0.01;


color = jet(round(1.5*Nneuron));


% plot_thresholds(rDec,Nneuron,color)
% hold off


% Example 0
% xT = 10*[1-cos(2*pi*t);2*pi*sin(2*pi*t)];
% dxT = 10*[2*pi*sin(2*pi*t);4*pi^2*cos(2*pi*t)];

% Example 1 : Rectangle
xT = [zeros(1,0.1*M),1/t(0.4*M)*t(1:0.4*M),ones(1,0.5*M)];
dxT = [zeros(1,0.1*M),1/t(0.4*M)*ones(1,0.4*M),zeros(1,0.5*M)];
% xT = [xT;dxT];
% dxT = [dxT;zeros(1,0.1*M-1),1/t(0.4*M)*1/dt,zeros(1,0.4*M-1),-1/t(0.4*M)*1/dt,zeros(1,0.5*M)];







% % Example 2
% xT = 200*sin(2*pi*t);
% dxT = 200*2*pi*cos(2*pi*t);



% Example 3 Step
xT = 10*[zeros(1,0.1*M),ones(1,0.9*M)];
dxT = 10*[zeros(1,0.1*M-1),1/dt,zeros(1,0.9*M)];
xT = xT + 0.1*randn(size(xT));
dxT = dxT + 2*randn(size(dxT));

% Example 4
%xT = [xT(1,:);xT(1,:)];
%dxT = zeros(2,M);

%Example 5 
% xT = 10*[-cos(2*pi*t)+1;2*pi*sin(2*pi*t)];
% dxT = 10*2*pi*[sin(2*pi*t);2*pi*cos(2*pi*t)];


% Example 6
t0 =8;
t1 =6;
a = 10;
xT1 = 1./(1+exp(-(a*t-t0)));
xT2 = 1./(1+exp(-(a*t-t1)));
dxT1 = a*exp(-(a*t-t0)).*xT1.^2;
dxT2 = a*exp(-(a*t-t1)).*xT2.^2;
aexp = exp((a*t-t0));
ddxT1 =(a.^2.*exp(a*t + t0).*(exp(t0) - exp(a*t)))./(exp(a*t) + exp(t0)).^3;
% dddxT1 = (3*a^3.*exp(t0 + a*t).*exp(a*t).*(exp(a*t) - exp(t0)))./(exp(a*t) + exp(t0)).^4 - (a^3.*exp(t0 + a*t).*(exp(a*t) - exp(t0)))./(exp(a*t) + exp(t0)).^3 - (a^3.*exp(t0 + a*t).*exp(a*t))./(exp(a*t) + exp(t0)).^3;
% ddddxT1 = -(a^4.*(exp(t0 + 4*a*t) - exp(4*t0 + a*t) - 11*exp(2*t0 + 3*a*t) + 11*exp(3*t0 + 2*a*t)))./(exp(5*t0) + 5*exp(t0 + 4*a*t) + exp(5*a*t) + 5*exp(4*t0 + a*t) + 10*exp(2*t0 + 3*a*t) + 10*exp(3*t0 + 2*a*t));
% % xT = 100*[1;0.5].*[xT1;xT2];
% % dxT = 100*[1;0.5].*[dxT1;dxT2];
% xT = [xT1;dxT1;ddxT1;dddxT1];
% dxT = [dxT1;ddxT1;dddxT1;ddddxT1];


lambdaD = 10;
iDec= 0.01*[1,-1,0,0;0,0,-1,1];
iDec = 0.05*[-ones(1,Nneuron/2),ones(1,Nneuron/2)];
rDec =00*iDec;


B = [0;1];
A = [0,1;-1,-5];
C = [1,1];


assert (B'*C' ~= 0,"You have chooses bad output/input matrices B'*C' =0")


phi = 0;
T = [cosd(phi), -sind(phi);
    sind(phi),cosd(phi)];



% A = [-0.31321,0.67575;0.1321,-7];

A = (T*A*inv(T));
B = T*B;
C = C*inv(T);


% A = -10;
% B = 1;
% C  = 1;

sys = ss(A,B,C,[]);
[num,den] = tfdata(sys);
num = cell2mat(num)
den = cell2mat(den)
% B = [0;1];
% A = [0,1;-fliplr(den(1,2:end))];
% C = fliplr(num(:,2:end));

%the matrix form is important!! 
%[xE,V,rate,spikes,error,feedforward,~,~,failed] = controller_without_error(A,B,C,xT,dxT,Nneuron,dt,M,rDec,iDec,mu,nu,lambdaD,sigmaV,lambdaV);
%[xE2,V,rate,spikes2,error,feedforward,~,~,failed2] = controller_without_error(A,B,C,xT,dxT,Nneuron,dt,M,rDec,iDec,mu,nu,lambdaD,sigmaV,lambdaV);
ALLOW_LEARNING = 1;
[xE,V,rate,spikes,error,feedforward,~,~,failed] = controller(A,B,C,xT,dxT,Nneuron,dt,M,rDec,iDec,mu,nu,lambdaD,sigmaV,lambdaV);

[xE,V,rate,spikes,error,feedforward,~,~,failed] = my_controller(A,B,C,xT,dxT,Nneuron,dt,M,rDec,iDec,mu,nu,lambdaD,sigmaV,lambdaV);

[err,l1,l2,L1,L2_squared] = calculateError(xT,C*xE,dt);

disp(sum(spikes,"all"))
disp(failed)

% f = figure();
% f.Position = [60,500,560,420];

p1 = plot (t,C*xE,"LineWidth",5,"DisplayName","network");
grid on
hold on
purpleColor = [128, 0, 128] / 255;
p2 = plot(t,xT,"LineStyle","--","Color",purpleColor,"LineWidth",5,"DisplayName","true");
p3 = plot(t,error,"r","DisplayName","error");
legend("show");
plot_spikes(t,spikes,80,0.5*max(xT,[],"all")/Nneuron,0,color);

return
display("Addition")
[err,l1,l2,L1,L2_squared] = calculateError(xT,xE2,dt);
disp(failed2);
disp(sum(spikes2,"all"))

% x changing the factor       Does affect error/spikes
% y changing the base         Doesnt affect error 
% z changing my change        Does affect both (for high lambdaD)
% u changing the lambdaD      Does affect the error/spikes without change
% Results error/ # spikes45

% plot x lambdaD y error
% plot x lambdaD y error with my change
% plot x factor,y base z error with my change
% just mention it changes the number of spikes 

clf
plot (t,C*xE,"LineWidth",3)
grid on
hold on
plot(t,xT,"LineWidth",3,"Color","k");
plot(t,xE2,"LineWidth",1);


