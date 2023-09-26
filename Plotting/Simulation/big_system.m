%test of the example
clear
close all
% Simulation
rng('default')
TE = 2.5; % sec
dt = 0.1e-3;
t = dt:dt:TE;
M = length(t);

getInput(t,10);

K = 10;
A = [zeros(3),eye(3);
    K*[-2,1,0;
       1,-2,1;
       0, 1,-1]./[1;2;3],zeros(3)];

c = [zeros(3,M);
    3-3*cos(3*t);
    1+4*sin(6*t);
    -1+2*sin(3*t);];

J = size(A,1);
x = zeros(J,M);
% Model
Nneuron= 200;% Neurons
lambdaD = 10; % Readout decay rate

while 1
    
Gamma = randn(J,Nneuron);
vn = vecnorm(Gamma,2,1);
Gamma = 1*0.03*Gamma./vn;

break
% Check that there are both inhibitory and exitory
% neurons
if all(any(Gamma < 0,2) == any(Gamma> 0,2))
    disp("Kernel found");
    disp("Exitory : "+ num2str(sum(Gamma >0)));
    disp("Inhibitory : " +num2str(sum(Gamma <0)));
    break
end
disp("Did not find a suitable random kernel. Try again.")
end

    
mu = 1*1e-6; % L2 Cost % Encourage spreading of work
nu = 0*1e-5; % L1 Cost % Penalize to many spikes
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


x = x(1:3,:);
xhat = xhat(1:3,:);
f = figure();

f.Position =[0,600,1300,700];
tiledlayout(3,1,'TileSpacing','none','Padding','Compact');

nexttile(1,[2,1]);

plot (t,xhat,"LineWidth",3)
grid on
hold on
plot(t,x,"LineWidth",3);
hold on

x2 = [t,fliplr(t)];
%inBetween = [x(1,:),fliplr(xhat)];
%fill(x2,inBetween,"r","FaceAlpha",0.2,"FaceColor","#e8a112","EdgeColor","none")
%plot(t,t*0 + threshold(1)/Gamma(1,1)+ x(1,:)+ 0*threshold(1),"LineWidth",2,"Color","#EDB120","LineStyle","--");
%plot(t,t*0 - threshold(1)/Gamma(1,1)+ x(1,:) + 0*threshold(1),"LineWidth",2,"Color","#EDB120","LineStyle","--");
%legend("Neural Network Simulation","Numeric Computation","interpreter", "latex","location","southeast","FontSize",30)
ax = gca;
ax.FontSize = 20;
set(ax,"XColor","none");
ylimit = ylim;
ytick = linspace(ylimit(1),ylimit(2),9);
ax.YTick =ytick(2:end);


nexttile
bx= gca;

plot_spikes(t,spikes,80,0.25,0,color)
grid on
set(bx,"YColor","none");
bx.FontSize = 20;
bx.YTick = [];
xlabel("Time","FontSize",30,"Interpreter","latex")



folder = "/home/max/Documents/University/Master Thesis/plots/Simulation/";
prompt = "File name: ";
name = input(prompt,"s");
dest = strcat(folder,name)
exportgraphics(f,dest,"ContentType","vector");
