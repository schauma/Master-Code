%test of the example
clear
close all
% Simulation
rng('default')
TE = 0.15; % sec
dt = 0.1e-3;
t = dt:dt:TE;
M = length(t);
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

A = -15*eye(2);

c = 2*[1/t(0.05*M)*t(1:0.05*M),ones(1,0.1*M),1-1/t(0.05*M)*t(1:0.05*M),zeros(1,0.8*M)];
c2 =1.5*[zeros(1,0.1*M),1/t(0.1*M)*t(1:0.1*M),ones(1,0.4*M),1-1/t(0.05*M)*t(1:0.05*M),zeros(1,0.35*M)]; 
c = 10*[c;c2];
c = 30*[3*exp(-20*t)+sin(20*t);
        -2*exp(-20*t) + sin(40*t)];

J = size(A,1);
x = zeros(J,M);
% Model
Nneuron= 4;% Neurons
lambdaD = 8; % Readout decay rate


   
Gamma = randn(J,Nneuron);
alpha = linspace(0,2*pi,Nneuron+1);alpha = alpha(1:end-1)';
F = [cos(alpha),sin(alpha)];
Gamma = F';


vn = vecnorm(Gamma,2,1);
Gamma = 10*0.03*Gamma./vn;

    
mu = 0*1e-6; % L2 Cost % Encourage spreading of work
nu = 6000*1e-5; % L1 Cost % Penalize to many spikes
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
% D=(rate'\x')';
% xhat2 = D*rate;


%[spikes,permutes]= sortrows(spikes,"descend");
%Gamma(:,1:end) = Gamma(:,permutes);



gridColor = 0.15*ones(4,1);
orange = 1/255*[255,160,0];
color = jet(round(1.5*Nneuron));
f = figure();

f.Position =[0,600,1200,1800];
tiledlayout(2,2,'TileSpacing','Compact','Padding','Compact');

nexttile(1,[1,2]);

plot (t,xhat,"LineWidth",3)
grid on
hold on
plot(t,x,"LineWidth",3);
hold on
plot_spikes(t,spikes,100,0.12,1,color)
ax = gca;
ax.FontSize = 20;
xlabel("Time","FontSize",30,"Interpreter","latex")
ax.YTick = -0.5:0.5:2.5;

xlim([0,TE])
rectangle1 = [0,-1,0.01,2.5];
rectangle("Position",rectangle1,"LineWidth",2,"EdgeColor","k")




ax = nexttile(3,[1,1]);

box1space = 1:rectangle1(3)/dt ;
tbox = t(box1space);
xlim(rectangle1([1,3]))
ylim(rectangle1([2,4]))
plot(tbox,xhat(:,box1space),"LineWidth",3)
hold on
plot(tbox,x(:,box1space),"LineWidth",3)
grid on
plot_spikes(tbox,spikes(:,box1space),100,0.12,0.25,color);
ax = gca;
ax.FontSize = 20;
xlabel("Time","FontSize",30,"Interpreter","latex")
ax.YTick = 0:0.5:2.5;
axis([tbox(1),tbox(end),-0.6,1.])
plot([0.008,0.008],[-10,10],"Color",[1,0,0,0.7],"LineStyle","--","LineWidth",3)

nexttile(4,[1,1]);
for i = Nneuron:-1:1
vec = Gamma(:,i);
normal = [-vec(2);vec(1)];

F = [vec(1),vec(2);-vec(2),vec(1)];
tp = F\[threshold(i);0];

AA =  tp + 0.7*normal;
BB =  tp - 0.7*normal;
hold on
plot([AA(1),BB(1)],[AA(2),BB(2)],"LineWidth",4,"Color",color(i,:),...
   "Marker",".","MarkerSize",4)

end
plot(0,0,"+","LineWidth",3,"MarkerSize",20,"MarkerEdgeColor","k");

plot([0,0],[-10,10],"Color",gridColor);
plot([-10,10],[0,0],"Color",gridColor);

axis([-0.3,0.3,-0.3,0.3]);


ax = gca;
xlabel("$\Delta x$","FontSize",30,"Interpreter","latex")
ylabel("$\Delta y$","FontSize",30,"Interpreter","latex")
ax.FontSize = 20;
ticks = -0.3:.15:0.3;
ax.XTick = ticks;
ax.YTick = ticks;


%Annotation

%Arrow 1
a = arrow([0;0],x(:,30),16,[],20,2);
a.EdgeColor = orange;
a.FaceColor = orange;
a.FaceAlpha = 0.5;
a.EdgeAlpha = 0.5;

%Arrow 2 
a = arrow(x(:,30),x(:,31)-xhat(:,31),16,[],20,2);
a.EdgeColor = color(1,:);
a.FaceColor = color(1,:);
%Arrow 3
a = arrow(x(:,31)-xhat(:,31),x(:,46)-xhat(:,46)-[0;0.01],16,[],20,2);
a.EdgeColor = orange;
a.FaceColor = orange;
a.FaceAlpha = 0.5;
a.EdgeAlpha = 0.5;

%Arrow 4
a = arrow(x(:,46)-xhat(:,46)-[0;0.01],x(:,47)-xhat(:,47),16,[],20,2);
a.EdgeColor = color(4,:);
a.FaceColor = color(4,:);

%Arrow 5
a = arrow(x(:,47)-xhat(:,47),1.05*(x(:,65)-xhat(:,65)),16,[],20,2);
a.EdgeColor = orange;
a.FaceColor = orange;
a.FaceAlpha = 0.5;
a.EdgeAlpha = 0.5;

%Arrow 6 
a = arrow(x(:,65)-xhat(:,65)+[0.015;0],x(:,66)-xhat(:,66)+ [0.015;0],16,[],20,2);
a.EdgeColor = color(1,:);
a.FaceColor = color(1,:);


%Arrow 7
a = arrow(x(:,66)-xhat(:,66),1.05*(x(:,80)-xhat(:,80)),16,[],20,2);
a.EdgeColor = orange;
a.FaceColor = orange;
a.FaceAlpha = 0.5;
a.EdgeAlpha = 0.5;
dd = x(:,80)-xhat(:,80);
pp =scatter(dd(1),dd(2),250,"MarkerFaceColor",[1,0,0]);
pp.MarkerFaceAlpha= 0.7;
pp.MarkerEdgeAlpha= 0.7;
pp.Marker = "square";
folder = "/home/max/Documents/University/Master Thesis/plots/Simulation/";
prompt = "File name: ";
name = input(prompt,"s");
dest = strcat(folder,name)
exportgraphics(f,dest,"ContentType","vector");
