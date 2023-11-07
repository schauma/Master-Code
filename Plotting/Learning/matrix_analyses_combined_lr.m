
newcolors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E","#77AC30","#4DBEEE",];
%colors = blue, orange,yellow,purple,green,turquoise,red

S =["combined_learning_lr_0.05.mat";
    "combined_learning_lr_0.01.mat";
    "combined_learning_lr_0.001.mat";
    ];

dr = [0.05,0.01,0.001];



max_errors=zeros(length(S),4000);
load("max_errors_lr.mat");

%range = [1,3,5];
range = 1:3;
S = S(range,:);
dr = dr(range);
max_errors = max_errors(range,:);
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

c = 25*[zeros(1,n_time);(1-exp(-4*t)).*sin(3*pi*t)];

c(:,60*pt:end) = 0;

Wf_true = round(-F'*F - mu*eye(n_neuron),9);
Ws_true = round(F'*(A+lambdaD*eye(J))*F,9);

[xE,xT,~,spikes,~]= simulate_network(A,c,F,Threshold,n_time,dt,...
    Ws_true,Wf_true,lambdaD,lambdaV);



x = 1:n_epochs;
eigs =zeros(n_neuron,n_epochs);
largest_rel_dev = zeros(2,n_epochs);
means = zeros(1,n_epochs);
stds = zeros(1,n_epochs);
max_error = zeros(1,n_epochs);
err = 0;
for i = 1:n_epochs

    mat_f = Wfs(:,:,i);
    mat_s = Wss(:,:,i);



%     Error
    % try
    % [xE,xT,~,~,~]= simulate_network(A,c,F,Threshold,n_time,dt,...
    % mat_s,mat_f,lambdaD,lambdaV);
    % max_error(i) = max(abs(xE-xT),[],"all");
    % catch exception
    %     err = err+1;
    %     max_error(i) = NaN;
    % end



    % Eigenvalues
    % eigs(:,i) = eig(mat_s);


    % Largest deviations
    dM = mat_f-Wf_true;
    dM2 = dM./Wf_true;
    dM2 = abs(dM2);
    [v,k] = max(dM2(~isinf(dM2)),[],"all","linear");
    largest_rel_dev(1,i) = dM2(k);

    dM = mat_s-Ws_true;
    dM2 = dM./Ws_true;
    dM2 = abs(dM2);
    [v,k] = max(dM2(~isinf(dM2)),[],"all","linear");
    largest_rel_dev(2,i) = dM2(k);


    % statistic deviations
    % stds(i)  = std(dM,1,"all");
    % means(i)= mean(abs(dM2),"all");


end
%max_errors(s,1:n_epochs)=max_error;
display(num2str(err) +"matrices didnt converge")
%semilogx(x,means)
%%

if(s == 1)
f = figure();
f.Position =[0,600,1300,700];
tt = tiledlayout(3,1,"TileSpacing","tight","Padding","none");
end

nexttile(1,[1,1])
loglog(x,largest_rel_dev(2,:),"LineWidth",3,"Marker",".","MarkerSize",5, "Color",newcolors(s));




hold on
ax = gca;
grid off
grid
ax.FontSize = 20;
xticklabels({});
xlim([1,n_epochs]);
yticks(10.^[-2:4])


nexttile(2,[1,1])
loglog(x,abs(max_errors(s,x)),"LineWidth",3,"Marker",".","MarkerSize",5,"Color",newcolors(s));


hold on
ax = gca;
grid off
grid
ax.FontSize = 20;
ax.XMinorGrid = "on";
xticklabels({});
yticks([1e-2,1e-1,1e0]);
xlim([1,n_epochs]);
ylim([1e-2,3]);


ax = nexttile(3,[1,1]);
semilogx(x,largest_rel_dev(1,:),"LineWidth",3,"Marker",".","MarkerSize",5, "Color",newcolors(s));

legends = [];
for le = 1:length(dr)
    format_text = sprintf("$\\alpha = %g$",dr(le));
    legends = [legends;format_text];
end
legend(legends,"FontSize",25,"Interpreter","latex","Location","northeast");

hold on
grid on
xlim([1,n_epochs]);
ax = gca;
%ax.YTick = ticks;
ax.FontSize = 20;
xlabel("\# Epoch","FontSize",30,"Interpreter","latex");

 
end


folder = "\\ug.kth.se\dfs\home\s\h\sharifs\appdata\xp.V2\Downloads\Master-Code-main\sendThis\";
prompt = "File name: ";
name = input(prompt,"s");
dest = strcat(folder,name)
exportgraphics(f,dest,"ContentType","vector");
