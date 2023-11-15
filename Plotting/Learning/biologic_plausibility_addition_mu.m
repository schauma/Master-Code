

TE = 1;
dt = 0.1e-3;
t= dt:dt:TE;
n_time = length(t);
n_training_time = 100000;
n_neuron = 50;
fast_learning_rate = 1*0.01;
slow_learning_rate = 1*0.001;
global beta1 beta2 Wf_true Ws_true
beta1 = 1.1;%1.9;
beta2 = 0.522;%0.53;
K = 1*10;
training_epochs =1000;
%There is a shady relation between K and lambdaD
lambdaD = 10;
lambdaV = 0*10;
mu = 1*1e-5;

% Model problem
% Damped oscilator
% External input is a bump function
%A = -10.*eye(2);
A = [0 , 1 ; -1, -10];

J = size(A,1);

%Very dependend on F
F = randn(J,n_neuron);
alpha = linspace(0,2*pi,n_neuron+1);alpha = alpha(1:end-1)';
F = [cos(alpha),sin(alpha)]';
%F = [-1;1];

vn= vecnorm(F,2,1);
F = 0.03*F./vn;

Threshold = (vecnorm(F,2,1)'.^2 + 1*mu)/2;

Wf_true = round(-F'*F - mu*eye(n_neuron),9);
Ws_true = round(F'*(A+lambdaD*eye(J))*F,9);


%Rerun the script multiple times and convergence gets better
%Until it becomes unstable
if ~exist("Wf","var")
    rng("default")
    Wf = 1e-4*randn(n_neuron);%*0*Wf_true;
end
if ~exist("Ws","var")
    Ws = 1e-4*randn(n_neuron);%0*Ws_true;
end
pt= 0.01*n_time;
c = 10*[(1-exp(-4*t)).*sin(3*pi*t);4*exp(-4*t).*sin(3*pi*t)+3*pi*(1-exp(-4*t)).*cos(3*pi*t)];
%c(:,60*pt:end) = 0;

%Learn with the parameters above
%%
Wf_save = Wf_learned;
Ws_save = Ws_learned;


%% Sim 1 
[xE,xT,rate,spikes,V]= simulate_network(A,c,F,Threshold,n_time,dt,...
    Ws_learned,Wf_learned,lambdaD,lambdaV);


 
%% Adjusted
Wf_adjusted = Wf_save - mu*eye(n_neuron);
display("Adjusted")
%% Sim 2 
[xE_a,xT_a,rate_a,spikes_a,V_a]= simulate_network(A,c,F,Threshold,n_time,dt,...
    Ws_learned,Wf_adjusted,lambdaD,lambdaV);

%% %Plotting
newcolors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E","#77AC30","#4DBEEE",];
%colors = blue, orange,yellow,purple,green,turquoise,red


f = figure();
f.Position =[0,600,1300,700];
tt = tiledlayout(3,1,"TileSpacing","tight","Padding","none");


nexttile(1,[1,1])

%Error between both approaches.
error = xE-xE_a;
error = abs(error/norm(F(:,1),2));
plot(t, error(1,:),"LineWidth",2,"Marker",".","MarkerSize",5, "Color",newcolors(1));
hold on
plot(t, error(2,:),"LineWidth",2,"Marker",".","MarkerSize",5, "Color",newcolors(2));

ax = gca;
grid on
ticks = -3:1:3;
ax.YTick = ticks;
xticklabels({});
ax.FontSize = 20;
legend(["$e_1$","$e_2$"],"FontSize",30,"Interpreter","latex")
ylabel({'Relative Spike';'Difference'},"FontSize",30,"Interpreter","latex")


nexttile(2,[1,1])

color = parula(75);
plot_spikes(t,spikes,25,1,-50,color);
ax =gca;
xticklabels({});
yticklabels({0,10,20,30,40,50})
grid on
tick = 0:10:50;
a.YTick = ticks;
ax.FontSize = 20;

ylabel("\# Neuron","FontSize",30,"Interpreter","latex")
ylim([0,50])
box on



nexttile(3,[1,1])
% Spikes with the adjustment
plot_spikes(t,spikes_a,25,1,-50,color);

ax =gca;
yticklabels({0,10,20,30,40,50})
grid on
tick = 0:10:50;
a.YTick = ticks;
ax.FontSize = 20;
xlabel("Time","FontSize",30,"Interpreter","latex");
ylabel("\# Neuron","FontSize",30,"Interpreter","latex");
ylim([0,50])
box on

%%


folder = "\\ug.kth.se\dfs\home\s\h\sharifs\appdata\xp.V2\Downloads\Master-Code-main\sendThis\";
prompt = "File name: ";
name = input(prompt,"s");
dest = strcat(folder,name)
exportgraphics(f,dest,"ContentType","vector");




