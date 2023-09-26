close all
clear variables
rng('shuffle')

Nneuron = 100;
dt  = 0.1e-3;
TE = 5; 
t = 0:dt:TE;
M = int64(TE/dt) + 1;

lambdaV = 1;
sigmaV = 1;
mu = 0.01;
nu = 0.1;


color = jet(round(1.5*Nneuron));
neuron_split_ration = Nneuron -floor(0.5*Nneuron);

% Example 0
xT = 10*[1-cos(2*pi*t);2*pi*sin(2*pi*t)];
dxT = 10*[2*pi*sin(2*pi*t);4*pi^2*cos(2*pi*t)];



lambdaD = 10;
iDec = 0.25*[-ones(1,Nneuron/2),ones(1,Nneuron/2)];
rDec =000*iDec;
B = [0;1];


A = [0,1;-5,-10];
C = eye(2);

[xE,V,rate,spikes,error,feedforward,~,~,failed] = controller_without_extra_term(A,B,C,xT,dxT,Nneuron,dt,M,rDec,iDec,mu,nu,lambdaD,sigmaV,lambdaV);

[err,l1,l2,L1,L2_squared] = calculateError(xT,xE,dt);

disp(sum(spikes,"all"))
disp(failed)

f = figure();

f.Position =[0,600,1300,700];
tiledlayout(3,1,'TileSpacing','none','Padding','Compact');

nexttile(1,[2,1]);

plot (t,xE(1,:),"LineWidth",3)
grid on
hold on
plot(t,xT(1,:),"LineWidth",3);
ax = gca;
ax.FontSize = 20;
set(ax,"XColor","none");
ylim([-10,30]);
ylimit = ylim;
ytick = linspace(ylimit(1),ylimit(2),9);
ax.YTick =ytick(2:end);


nexttile
bx= gca;

bounds = plot_spikes(t,spikes,80,1,0,color);
grid on
set(bx,"YColor","none");
bx.FontSize = 20;
bx.YTick = [];
xlabel("Time","FontSize",30,"Interpreter","latex")
xlim([0,TE]);
xlimit  = xlim;
xtick = linspace(xlimit(1),xlimit(end),4);
bx.XTick = xtick(1:end);
ylim(bounds);

folder = "/home/max/Documents/University/Master Thesis/plots/Control/";
prompt = "File name: ";
name = input(prompt,"s");
dest = strcat(folder,name)
exportgraphics(f,dest,"ContentType","vector");


