load("Wf_learning_5000.mat");
load("Ws_learning_5000.mat");

n_epochs = size(Wfs,3);
n_neuron = size(Wfs,1);

alpha = linspace(0,2*pi,n_neuron+1);alpha = alpha(1:end-1)';
F = [cos(alpha),sin(alpha)]';
vn= vecnorm(F,2,1);
mu = 1*1e-5;
lambdaD = 10;
lambdaV = 0;
F = 0.03*F./vn;
A = [0 , 1 ; -1, -10];
J = size(A,1);
Threshold = (vecnorm(F,2,1)'.^2 + mu)/2;



TE = 1;
dt = 0.1e-3;
t= dt:dt:TE;
n_time = length(t);
pt= 0.01*n_time;
c = 25*[zeros(1,n_time);(1-exp(-4*t)).*sin(3*pi*t)];
c(:,60*pt:end) = 0;

Wf_true = round(-F'*F - mu*eye(n_neuron),9);
Ws_true = round(F'*(A+lambdaD*eye(J))*F,9);

[xE,xT,~,spikes,~]= simulate_network(A,c,F,Threshold,n_time,dt,...
    Ws_true,Wf_true,lambdaD,lambdaV);



close all
f = figure();
f.Position =[0,600,1300,700];
tiledlayout(3,1,"TileSpacing","tight","Padding","none")
nexttile

plot(t,c(2,:),"LineWidth",5,"Marker",".","MarkerSize",10);
ax = gca;
grid on
ax.FontSize = 20;
%ylim([-0.5,11])
xticklabels({});
legend("c","FontSize",30,"Interpreter","latex");

ax = nexttile(2,[2,1]);
plot(t,xT,"LineWidth",5);
hold on;
plot(t,xE,"LineWidth",3);
grid on

ax = gca;
ticks = -2:.5:3;
ylim([-2,1.5]);
ax.YTick = ticks;
ax.FontSize = 20;
xlabel("Time","FontSize",30,"Interpreter","latex");
legend(["","","$x_1$","$x_2$"],"FontSize",30,"Interpreter","latex")

figure
plot_spikes(t,spikes,80,1,0,jet(50));
folder = "/home/max/Documents/University/Master Thesis/plots/Learning/";
prompt = "File name: ";
name = input(prompt,"s");
dest = strcat(folder,name)
exportgraphics(f,dest,"ContentType","vector");

