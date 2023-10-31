

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
c = 10*[zeros(J,15*pt),ones(J,30*pt),0*-1*ones(J,30*pt),zeros(J,25*pt)];


Wf_true = round(-F'*F - mu*eye(n_neuron),9);
Ws_true = round(F'*(A+lambdaD*eye(J))*F,9);

[xE,xT,~,~,~]= simulate_network(A,c,F,Threshold,n_time,dt,...
    Ws_true,Wf_true,lambdaD,lambdaV);

%%

close all
f = figure();
f.Position =[0,600,1300,700];
tiledlayout(3,1,"TileSpacing","tight","Padding","none")
nexttile

plot(t,c(2,:),"LineWidth",5,"Marker",".","MarkerSize",10);
ax = gca;
grid on
ax.FontSize = 20;
ylim([-0.5,11])
xticklabels({});
legend("c","FontSize",30,"Interpreter","latex");

ax = nexttile(2,[2,1]);
plot(t,xT,"LineWidth",5);
hold on;
plot(t,xE,"LineWidth",3);
grid on

ax = gca;
ticks = -0.5:.5:3;
ax.YTick = ticks;
ax.FontSize = 20;
xlabel("Time","FontSize",30,"Interpreter","latex");
legend(["","","$x_1$","$x_2$"],"FontSize",30,"Interpreter","latex")

% folder = "/home/max/Documents/University/Master Thesis/plots/Learning/";
% prompt = "File name: ";
% name = input(prompt,"s");
% dest = strcat(folder,name)
% exportgraphics(f,dest,"ContentType","vector");


%%
x = 1: n_epochs;
eigs =zeros(n_neuron,n_epochs);
largest_rel_dev = zeros(1,n_epochs);
means = zeros(1,n_epochs);
stds = zeros(1,n_epochs);
max_errors = zeros(1,n_epochs);
err = 0;
for i = 1:n_epochs

    mat = Wfs(:,:,i);
    true_matrix = Wf_true;



    % Error
%     try
%     [xE,xT,~,~,~]= simulate_network(A,c,F,Threshold,n_time,dt,...
%     Ws_true,mat,lambdaD,lambdaV);
%     max_errors(i) = max(abs(xE-xT),[],"all");
%     catch exception
%         err = err+1;
%         max_errors(i) = NaN;
%     end

    

    % Eigenvalues
    eigs(:,i) = eig(mat);


    % Largest deviations
    dM = mat-true_matrix;
    dM2 = dM./true_matrix;
    dM2 = abs(dM2);
    [v,k] = max(dM2(~isinf(dM2)),[],"all","linear");
    largest_rel_dev(i) = dM2(k);



    % statistic deviations
    s = std(dM,1,"all");
    stds(i)  = s;
    means(i)= mean(abs(dM2),"all");


end
display(num2str(err) +"matrices didnt converge")


%%
semilogx(x,means)

%%


f = figure();
f.Position =[0,600,1300,700];
tiledlayout(3,1,"TileSpacing","tight","Padding","none")
nexttile(2,[1,1])
semilogx(x,largest_rel_dev,"LineWidth",3,"Marker",".","MarkerSize",5)

ax = gca;
grid on
xlim([0,6000]);
ax.FontSize = 20;
xticklabels({});

%legend("Largest rel. Deviation","FontSize",30,"Interpreter","latex");


nexttile(1,[1,1])
semilogx(x,eigs(1,:),"LineWidth",5,"Marker",".","MarkerSize",10)
hold on
semilogx(x,eigs(2,:),"LineWidth",5,"Marker",".","MarkerSize",10)
hold on
semilogx([1,6000],[-0.024,-0.024],"Color",[0.9290 0.6940 0.1250])
text(1.5,-0.024+0.0003,"optimal","FontSize",20,"Color",...
    [0.9290 0.6940 0.1250],"Interpreter","latex",...
    "EdgeColor",[0.9290 0.6940 0.1250],"LineWidth",1,...
    "VerticalAlignment","bottom");
ax = gca;
grid on
xlim([0,6000]);
ax.FontSize = 20;
xticklabels({});
legend(["$\lambda1$","$\lambda_2$",""],"FontSize",30,"Interpreter","latex")


ax = nexttile(3,[1,1]);
semilogx(x,max_errors,"LineWidth",3,"Marker",".","MarkerSize",5)

grid on
ax = gca;
xlim([0,6000]);
%ax.YTick = ticks;
ax.FontSize = 20;
xlabel("\# Epoch","FontSize",30,"Interpreter","latex");



%%

[xE,xT,rate,spikes1630,V]= simulate_network(A,c,F,Threshold,n_time,dt,...
    Ws_true,Wfs(:,:,1630),lambdaD,lambdaV);


[xE,xT,rate,spikes724,V]= simulate_network(A,c,F,Threshold,n_time,dt,...
    Ws_true,Wfs(:,:,724),lambdaD,lambdaV);

bar(sum(spikes1630'))
title("1630")
figure
bar(sum(spikes724'))
title("724")

%%
minimum = 0.9003197828372014;
maximum = 1.1194354976378833; 
f = figure(9);
f.Position =[0,600,1300,700];

tiledlayout(1,2,"TileSpacing","tight","Padding","none")
nexttile(1,[1,1])

imagesc(Wfs(:,:,1630)-Wf_true)
set(gca,'XAxisLocation','top');
set(gca,"YAxisLocation","right");
title("1630")
ax = gca;
yticklabels({});
colormap jet
cb = colorbar;
%clim([minimum,maximum]);
cb.Location = "westoutside";
ax.FontSize = 20;


nexttile(2,[1,1])
imagesc(Wfs(:,:,724)-Wf_true)
set(gca,'XAxisLocation','top');
colorbar;
title("724")
colormap jet
clim([-2e-5,2e-5]);
ax = gca;
ax.FontSize = 20;
folder = "/home/max/Documents/University/Master Thesis/plots/Learning/";
prompt = "File name: ";
name = input(prompt,"s");
dest = strcat(folder,name)
exportgraphics(f,dest,"ContentType","vector");


