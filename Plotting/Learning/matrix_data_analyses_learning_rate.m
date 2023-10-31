clear
newcolors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E","#77AC30","#4DBEEE",];
% colors = blue , orange,yellow,purple,green,turquoise,red

S = ["Wf_lr_0.0001";
    %"Wf_lr_0.001";
    "Wf_lr_0.0005";
    "Wf_lr_0.005"
    ];
load("max_errors_lr.mat")

max_errors = max_errors([1,2,4],:);

S = flipud(S);
max_errors  = flipud(max_errors);

for s = 1:length(S)

load(S(s));

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



x = 1:n_epochs;
eigs =zeros(n_neuron,n_epochs);
largest_rel_dev = zeros(1,n_epochs);
means = zeros(1,n_epochs);
stds = zeros(1,n_epochs);
max_error = zeros(1,n_epochs);
err = 0;
for i = 1:n_epochs

    mat = Wfs(:,:,i);
    true_matrix = Wf_true;



    % Error
%     try
%     [xE,xT,~,~,~]= simulate_network(A,c,F,Threshold,n_time,dt,...
%     Ws_true,mat,lambdaD,lambdaV);
%     max_error(i) = max(abs(xE-xT),[],"all");
%     catch exception
%         err = err+1;
%         max_error(i) = NaN;
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
    stds(i)  = std(dM,1,"all");
    means(i)= mean(abs(dM2),"all");


end

display(num2str(err) +"matrices didnt converge")
%semilogx(x,means)
%%

if(s == 1)
f = figure();
f.Position =[0,600,1300,700];
tt = tiledlayout(2,1,"TileSpacing","tight","Padding","none");
end

nexttile(1,[1,1])
semilogx(x,largest_rel_dev,"LineWidth",3,"Marker",".","MarkerSize",5)
hold on
ax = gca;
grid on
ax.FontSize = 20;
xticklabels({});
legend(["$\alpha = 0.005$","$\alpha = 0.0005$","$\alpha = 0.0001$"],"FontSize",30,"Interpreter","latex");

ax = nexttile(2,[1,1]);
semilogx(1:4000,max_errors(s,:),"LineWidth",3,"Marker",".","MarkerSize",5)
ylim([0,3])
hold on
grid on
ax = gca;
%ax.YTick = ticks;
ax.FontSize = 20;
xlabel("\# Epoch","FontSize",30,"Interpreter","latex");


% ax = nexttile(3,[1,1]);
% semilogx(x,eigs(1:2,:),"LineWidth",3,"Marker",".","MarkerSize",5,"Color",newcolors(s))
% hold on
% grid on
% ax = gca;
% %ax.YTick = ticks;
% ax.FontSize = 20;
% xlabel("\# Epoch","FontSize",30,"Interpreter","latex");

end


folder = "/home/max/Documents/University/Master Thesis/plots/Learning/";
prompt = "File name: ";
name = input(prompt,"s");
dest = strcat(folder,name)
exportgraphics(f,dest,"ContentType","vector");


