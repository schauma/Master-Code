%clear variables
%close all
TE = 1;
dt = 0.1e-3;
t= dt:dt:TE;
n_time = length(t);
n_training_time = 100000;
n_neuron = 50;
fast_learning_rate = 1*0.01;
slow_learning_rate = 1*0.001;
global beta1 beta2 Wf_true Ws_true
beta = 1;
beta1 = 1.90;%1.9;
beta2 = 0.522;%0.53;
K = 1*10;
training_epochs =4000;
%There is a shady relation between K and lambdaD
lambdaD = 10;
lambdaV = 0*10;
mu = 1*1e-5;

% Model problem
% Damped oscilator
% External input is a bump function
%A = -1.*eye(1);
A = [0 , 1 ; -1, -10];
J = size(A,1);

%Very dependend on F
F = randn(J,n_neuron);
alpha = linspace(0,2*pi,n_neuron+1);alpha = alpha(1:end-1)';
F = [cos(alpha),sin(alpha)]';
%F = [-1;1];



vn= vecnorm(F,2,1);
F = 0.03*F./vn;

Threshold = (vecnorm(F,2,1)'.^2 + mu)/2;

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
c = 10*[zeros(J,15*pt),ones(J,30*pt),1*-1*ones(J,30*pt),zeros(J,25*pt)];
c = 10*[(1-exp(-4*t)).*sin(3*pi*t);4*exp(-4*t).*sin(3*pi*t)+3*pi*(1-exp(-4*t)).*cos(3*pi*t)];

% Adjustments!!!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Big system:
% A = [zeros(3),eye(3);
%     K*[-2,1,0;
%        1,-2,1;
%        0, 1,-1]./[1;2;3],zeros(3)];
% 
% c = [zeros(3,length(t));
%     3-3*cos(3*t);
%     1+4*sin(6*t);
%     -1+2*sin(3*t);];

% J = size(A,1);
F  = randn(J,n_neuron);
vn= vecnorm(F,2,1);
F = 0.03*F./vn;
Threshold = (vecnorm(F,2,1)'.^2 + mu)/2;
Wf_true = round(-F'*F - mu*eye(n_neuron),9);
Ws_true = round(F'*(A+lambdaD*eye(J))*F,9);
Ws = Ws_true;
slow_learning_rate = 0;
fast_learning_rate = 0.005;
Wf = 0*Wf_true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[xE_true,xT,rate,spikes,V]= simulate_network(A,c,F,Threshold,n_time,dt,...
    Ws_true,Wf_true,lambdaD,lambdaV);

% Ping Ponging effect!
[Wf_learned, Ws_learned,Wfs,Wss] = bourdoukan_learning(A,F,Threshold-mu/2,...
    n_training_time,dt,Ws,Wf,lambdaD,lambdaV,fast_learning_rate,...
    slow_learning_rate,training_epochs,mu,beta,K);
Ws = Ws_learned;
Wf = Wf_learned;

[xE,xT,rate,spikes,V]= simulate_network(A,c,F,Threshold-mu/2,n_time,dt,...
    Ws_learned,Wf_learned,lambdaD,lambdaV);



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

function [Wf,Ws,Wfs,Wss] = bourdoukan_learning(A,F,Threshold,n_time,dt,Ws,Wf,...
    lambdaD,lambdaV,fast_learning_rate,slow_learning_rate,n_epochs,mu,beta,K)
global beta1 beta2 Ws_true Wf_true

slow_drop_rate = 0*0.03;
fast_drop_rate =0.09;%compare wf with .7 and .5 drop rate.what is the critical change?

lr_base_fast = fast_learning_rate;
lr_base_slow = slow_learning_rate;


J = size(A,1);
n_neuron = size(Ws,1);
V = zeros(n_neuron,1);
rate = V;
xT = zeros(J,1);
xE = xT;
one_hot = eye(n_neuron);


Wss = zeros([n_neuron,n_neuron,n_epochs]);
Wfs = Wss;


sigma=abs(60); %std of the smoothing kernel

w=(1/(sigma*sqrt(2*pi)))* exp(-(([1:n_time]-n_time/2).^2)/(2*sigma.^2));%gaussian smoothing kernel used to smooth the input
w=w/sum(w); % normalization of the kernel



for ee = 1:n_epochs

    c = mvnrnd(zeros(1,J),eye(J),n_time)';
    %c(:,1:70)= 0;
    for d=1:J
        c(d,:)=0.005/dt*conv (c(d,:),w,'same'); %smoothing the previously generated white noise with the gaussian window w
    end
    xE = 0*xE;
    xT = 0*xT;
    rate = 0*rate;
    V = 0*V;
    for i = 1:n_time-1

        e = xT - xE;
        V = (1-lambdaV*dt)*V...
            + dt*F'*c(:,i)...
            + dt*Ws*rate...
            + dt*K*F'*e...
            + 0.000*randn(n_neuron,1);



        [val,idx] = max(V-Threshold);
        while val>0

            % Learning

            % Fast learning
            %dWf = (V+ mu*rate)*beta1+ Wf(:,idx) + mu*one_hot(:,idx);
            dWf = V + beta2*Wf(:,idx);
            Wf(:,idx) = Wf(:,idx) - fast_learning_rate*dWf;

             


            % Slow learning
            e = xT - F*rate;

            dM = e*(F*rate)';
            % The random addache of that term helps a bit
            dWs = F'*dM*F;
            Ws = Ws + slow_learning_rate*dWs;



            %Process spike
            V = V + Wf(:,idx);

            %Adjust rate
            rate(idx) = rate(idx)+1;


            break;
            [val,idx] = max(V-Threshold);
        end

        rate = (1-lambdaD*dt)*rate;
        xE = F*rate;
        xT = (eye(J)+A*dt)*xT +dt*c(:,i+1);

    end

    
    slow_learning_rate = (1-slow_drop_rate)*slow_learning_rate;
    fast_learning_rate = ((1-fast_drop_rate)^ee + 0.02)*lr_base_fast;

    Wss(:,:,ee) = Ws;
    Wfs(:,:,ee) = Wf;


    disp("Learning epoch: "+ num2str(ee) + "/" + num2str(n_epochs));
    Wf_ratio = Wf./Wf_true;
    Ws_ratio = Ws./Ws_true;

    dWf = Wf-Wf_true;
    dWf = abs(dWf./Wf_true);

    display(max(dWf(~isinf(dWf)),[],"all"))



    dWs = Ws-Ws_true;
    dWs = abs(dWs./Ws_true);

    display(max(dWs(~isinf(dWs)),[],"all"))

end
end



