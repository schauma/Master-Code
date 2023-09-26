%test of the example
clear
close all
% Simulation
rng('default')
TE = 2.5; % sec
dt = 0.1e-2;
t = dt:dt:TE;
M = length(t);

%getInput(t,10);
% K = 10;
% A = [zeros(3),ones(3);
%     K*[-2,1,0;
%        1,-2,1;
%        0, 1,-1],zeros(3)];
% 
% c = [zeros(3,M);
%     5+5*cos(3*t);
%     3+3*sin(3*t);
%     -1-1*sin(3*t);];

A = -10*eye(2);

c = 2*[1/t(0.1*M)*t(1:0.1*M),ones(1,0.05*M)];
c2 =0*[1/t(0.1*M)*t(1:0.1*M),ones(1,0.05*M)]; 
c = 10*[c,2*cos(3*t(1:0.895*M));c2,2*sin(3*t(1:0.895*M))];
J = size(A,1);
x = zeros(J,M);
% Model
Nneuron= 15;% Neurons
lambdaD =5; % Readout decay rate

alpha = linspace(0,2*pi,Nneuron+1);alpha = alpha(1:end-1)';
F = [cos(alpha),sin(alpha)];
%Gamma = randn(J,Nneuron);
Gamma = F';
vn = vecnorm(Gamma,2,1);
Gamma = 10*0.03*Gamma./vn;
    
mu = 0*1e-3; % L2 Cost % Encourage spreading of work
nu = 1*1e-2; % L1 Cost % Penalize to many spikes
lambdaV =0*10; % Leak Voltage term
sigmaV = 0*1000*1e-3; %noise for Voltage equation

threshold = (nu + mu + vecnorm(Gamma,2,1).^2)'/2; % Threshold for each neuron

OmegaS = Gamma'*(A+lambdaD*eye(J))*Gamma; % Slow Dynamics
OmegaF = -Gamma'*Gamma - mu*eye(Nneuron,Nneuron);
spikes = zeros(Nneuron,M); % Keeps track at what time step a neuron spikes
rate = zeros(Nneuron,M);




% Simulation
V= zeros(Nneuron,M);


% Now voltage equation. Compute Voltage, if voltage higher than
% threshold, fire a spike, add spike to list.

for i = 1:M-1
    % Reference
    x(:,i+1) = x(:,i) + dt*A*x(:,i) + dt.*c(:,i);

    noise = randn(Nneuron,1);
    V(:,i+1) = (1-lambdaV*dt)*V(:,i)...
        + dt*Gamma'*c(:,i) ...
        + dt*OmegaS*rate(:,i) ...
        + 0*OmegaF*spikes(:,i)...
        + dt*sigmaV*noise;

    [val,idx]= max(V(:,i+1) - threshold);%finding the neuron with largest membrane potential
    
    while val > 0
        spikes(idx,i+1) = spikes(idx,i+1) + 1;
        V(:,i+1) = V(:,i+1) + OmegaF(:,idx);
        
        
        [val,idx] = max(V(:,i+1)-threshold);
    end
    if (val>0)  %if its membrane potential exceeds the threshold the neuron k spikes
        spikes(idx,i+1)=1; % the spike ariable is turned to one
    end

    rate(:,i+1)=(1-lambdaD*dt)*rate(:,i)+spikes(:,i+1); %filtering the spikes
end

xhat =Gamma*rate;
D=(rate'\x')';
xhat2 = D*rate;


color = jet(round(1.5*Nneuron));
f = figure();

f.Position =[0,600,1400,450];
tiledlayout(1,3,'TileSpacing','Compact','Padding','Compact');

nexttile(1,[1,2]);

plot (t,xhat,"LineWidth",3)
grid on
hold on
plot(t,x,"LineWidth",3);
hold on
plot_spikes(t,spikes,100,0.25,0,color)
%inBetween = [x(1,:),fliplr(xhat)];
%fill(x2,inBetween,"r","FaceAlpha",0.2,"FaceColor","#e8a112","EdgeColor","none")
%plot(t,t*0 + threshold(1)/Gamma(1,1)+ x(1,:)+ 0*threshold(1),"LineWidth",2,"Color","#EDB120","LineStyle","--");
%plot(t,t*0 - threshold(1)/Gamma(1,1)+ x(1,:) + 0*threshold(1),"LineWidth",2,"Color","#EDB120","LineStyle","--");
%legend("Neural Network Simulation","Numeric Computation","interpreter", "latex","location","southeast","FontSize",30)
ax = gca;
ax.FontSize = 20;
xlabel("Time","FontSize",30,"Interpreter","latex")
ax.YTick = 0:0.5:2.5;

xlim([0,2.5])

nexttile

sspikes = sum(spikes,2);
a =bar(1:Nneuron,sspikes);
a.FaceColor = "flat";
a.CData = color(1:Nneuron,:);
txt = sprintf("In Total: %d spikes",sum(sspikes));
title(txt,"FontSize",30,"Interpreter","latex");
ax = gca;
ax.FontSize = 20;
xlabel("Neuron","FontSize",30,"interpreter","latex");




folder = "/home/max/Documents/University/Master Thesis/plots/Simulation/";
prompt = "File name: ";
name = input(prompt,"s");
dest = strcat(folder,name)
exportgraphics(f,dest,"ContentType","vector");
