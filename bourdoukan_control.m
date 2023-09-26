clear variables
close all
rng("default")
TE = 1;
dt = 0.1e-3;
t= dt:dt:TE;
n_time = length(t);
n_training_time = 10000;
n_neuron = 10;
fast_learning_rate = 0.01;
slow_learning_rate = 0.1;
global beta1 beta2 Wf_true Ws_true
beta = 1;
beta1 = 1.9;
beta2 = 0.51;
K = 10;
training_epochs =1000;
%There is a shady relation between K and lambdaD
lambdaD = 10;
lambdaV = 10;
sigmaV = 0*10e-3;
mu = 100*1e-6;

% Model problem
% Damped oscilator
% External input is a bump function
%A = -1.*eye(1);
A = 50*[0,1 ; -1, -2];
J = size(A,1);

%Very dependend on F
F = randn(n_neuron,J);
alpha = linspace(0,2*pi,n_neuron+1);alpha = alpha(1:end-1)';
F = [cos(alpha),sin(alpha)];
%F = [-1;1];
vn= vecnorm(F,2,2);
F = 0.03*F./vn;


Threshold = (vecnorm(F,2,2).^2 + mu)/2;


clc
% F = F/0.03;
Wf_true = 1*round(-F*F' - mu*eye(n_neuron),9);
Ws_true = 1*round(F*(A+lambdaD*eye(J))*F',9);
%%
close all
figure
quiver(zeros(n_neuron,1),zeros(n_neuron,1),F(:,1),F(:,2),1/norm(F(:,1)))
hold on
grid on
viscircles([0,0],1);
[U,S,V] = svd(F);
[V,D] = eig(A+lambdaD*eye(J),inv(F'*F));
[~,D2] = eig(Ws_true)
hold off

2*D/(0.03^2*n_neuron)
%D2
eig(A+eye(J)*lambdaD)

%%
Wf = 1*Wf_true;
Ws = 0*Ws_true;
%Ws = 0.01*randn(n_neuron,n_neuron)
pt= 0.01*n_time;
c = 5*[zeros(J,15*pt),ones(J,30*pt),-1*ones(J,30*pt),zeros(J,25*pt)];
[xE_true,xT,rate,spikes,V]= simulate_network(A,c,F',Threshold,n_time,dt,...
     Ws_true,Wf_true,lambdaD,lambdaV);

% Ping Ponging effect!
[Wf_learned, Ws_learned,Ws_tensor] = bourdoukan_learning(A,F,Threshold,...
   n_training_time,dt,Ws,Wf,lambdaD,lambdaV,fast_learning_rate,...
   slow_learning_rate,training_epochs,mu,beta,K);
Ws = Ws_learned;
Wf = Wf_learned;



[xE,xT,rate,spikes,V]= simulate_network(A,c,F',Threshold,n_time,dt,...
    Ws_learned,Wf_learned,lambdaD,lambdaV);

%%
TE = 50;
t = 0:dt:TE;
n_time = length(t);
% run "new" control simulation with direct appproach
clf
omega = 5;
ref = [sin(t)+1 + 0.5*sin(omega*t);
       (cos(t) + 0.5*omega*cos(omega*t))];
plot(t,ref)
%%
clf
[xE,xT,rate,spikes,V] = sim_net(A,ref,F,Threshold,n_time,dt,...
    Ws,Wf,lambdaD,lambdaV,sigmaV);



%%
Ds = zeros(J,101);
for ji = 1:101
D = sort(eig(squeeze(Ws_tensor(:,:,ji))),"descend")
Ds(:,ji) = D(1:2);
end

plot(1:101,Ds)
hold on
D_true = eig(A+lambdaD*eye(J), inv(F'*F));
plot(1:101,ones(1,101).*D_true)

%%
figure
tlo = tiledlayout(n_neuron,n_neuron,'TileSpacing','none','Padding','none'); 
for pici = 0:n_neuron^2-1
nexttile;
r = fix(pici/n_neuron)+1;
c =mod(pici,n_neuron)+1;
plot(1:101,squeeze(Ws_tensor(r,c,:)./Ws_true(r,c)));
hold on
plot(1:101,squeeze(Ws_tensor_add(r,c,:)),"Color","r");

ylim([0,1])

end

%set(tlo.Children,"XTick",[],"YTick",[]);
%%
figure
plot(t,xT,"Color","blue")
hold on
plot(t,xE,"Color","red")
plot(t,xE_true,"Color","green")
%legend("$x$","$\dot{x}$","FontSize",15,"interpreter","latex")
%plot_spikes(t,spikes)
hold off

figure
plot(t,vecnorm(xT-xE,2,1))
title("Error over time")
legend("$||x - \hat{x}||$","Interpreter","latex","FontSize",20)
% figure
% plot(t,vecnorm(rate,Inf,1))
% legend("Firing rates")

function [Wf,Ws,Ws_tensor] = bourdoukan_learning(A,F,Threshold,n_time,dt,Ws,Wf,...
    lambdaD,lambdaV,fast_learning_rate,slow_learning_rate,n_epochs,mu,beta,K)
global beta1 beta2 Ws_true Wf_true

slow_drop_rate = 0*0.03;
fast_drop_rate = 0*0.05;
completed_learning  = 1;



J = size(A,1);
n_neuron = size(Ws,1);
V = zeros(n_neuron,1);
spikes = V;
rate = V;
xT = zeros(J,1);
xE = xT;


sigma=abs(60); %std of the smoothing kernel
w=(1/(sigma*sqrt(2*pi)))* exp(-(([1:n_time]-n_time/2).^2)/(2*sigma.^2));%gaussian smoothing kernel used to smooth the input
w=w/sum(w); % normalization of the kernel


Ws_tensor  = zeros(n_neuron,n_neuron,100);


for ee = 1:n_epochs

    c = mvnrnd(zeros(1,J),eye(J),n_time)';
    c(:,1:70)= 0;
    for d=1:J
        c(d,:)=0.01/dt*conv(c(d,:),w,'same'); %smoothing the previously generated white noise with the gaussian window w
    end
    xE = 0*xE;
    xT = 0*xT;
    rate = 0*rate;
    V = 0*V;
    for i = 1:n_time-1
        
        e = xT - xE;
        V = (1-lambdaV*dt)*V...
            + dt*F*c(:,i)...
            + dt*Ws*rate...
            + 0*Wf*spikes...
            + dt*K*F*e...
            + 0.000*randn(n_neuron,1);

        
        
        [val,idx] = max(V-Threshold);
        spikes = 0*spikes;
        while val>0
            rate(idx) = rate(idx)+1;
            % Learning            
            
            % Fast learning
            %dWf = -((V+ mu*rate)*beta1+ Wf(:,idx) + mu*one_hot(n_neuron,idx));
            dWf = -(V + beta2*Wf(:,idx));
            if i<0.7*n_time
            %Wf(:,idx) = Wf(:,idx) + fast_learning_rate*dWf;
            end
            
            % Slow learning
            e = xT - F'*rate;
            dM = e*(rate'*F);
            % The random addache of that term helps a bit
            dWs = F*dM*F'+0*0.0001*lambdaD*(F*F');
            
            Ws = Ws + slow_learning_rate*dWs;
            %Process spike
            V = V + Wf(:,idx);

            [val,idx] = max(V-Threshold);
        end

        rate = (1-lambdaD*dt)*rate;
        xE = F'*rate;
        xT = (eye(J)+A*dt)*xT +dt*c(:,i+1);
        
    end
    

    if ee/n_epochs >= completed_learning/100
        slow_learning_rate = (1-slow_drop_rate)*slow_learning_rate;
        fast_learning_rate = (1-fast_drop_rate)*fast_learning_rate;
        
        disp("Learning: "+ num2str(completed_learning) +"%")
        completed_learning = completed_learning+1;
        Wf_ratio = Wf./Wf_true;
        Ws_ratio = Ws./Ws_true;
        Ws_tensor(:,:,completed_learning) = Ws;
    end

end
end

function v = one_hot(N,k)
v = zeros(N,1);
v(k) =1;

end



function [xE,xT,rate,spikes,V] = sim_net(A,ref,F,Threshold,n_time,dt,...
    Ws,Wf,lambdaD,lambdaV,sigmaV)
J = size(A,1);
n_neuron = size(Ws,1);

V = zeros(n_neuron,n_time);
spikes = zeros(n_neuron,n_time);
rate = zeros(n_neuron,n_time);
xE = zeros(J,n_time);
xT = xE;
c = zeros(J,n_time);
for i = 1:n_time-1
    noise = randn(n_neuron,1);
    xT(:,i+1) = (eye(J)+A*dt)*xT(:,i) +dt*c(:,i);
    V(:,i+1) = (1-lambdaV*dt)*V(:,i)...
        + dt*F*c(:,i)...
        + dt*Ws*rate(:,i)...
        + dt*sigmaV*noise;


    [val,idx] = max(V(:,i+1)-Threshold);

    while val >0
        spikes(idx,i+1) = spikes(idx,i+1) + 1;
        V(:,i+1) = V(:,i+1) + Wf(:,idx);


        [val,idx] = max(V(:,i+1)-Threshold);
    end

    rate(:,i+1) = (1-lambdaD*dt)*rate(:,i) + spikes(:,i+1);
    xE(:,i+1) = F'*rate(:,i+1);
    %From controller papers
    c(:,i+1) = (ref(:,i+1)-ref(:,i))/dt - A*ref(:,i+1);
    % Idea
    %c(:,i+1) =
    % ref(:,i+1) - xE(:,i+1);
    % d = c(:,i+1)
    
end
plot(dt*(1:n_time),xE,dt*(1:n_time),ref)
end
