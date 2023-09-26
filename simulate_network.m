function [xE,xT,rate,spikes,V] = simulate_network(A,c,F,Threshold,n_time,dt,Ws,Wf,...
    lambdaD,lambdaV)

J = size(A,1);
n_neuron = size(Ws,1);

V = zeros(n_neuron,n_time);
spikes = zeros(n_neuron,n_time);
rate = zeros(n_neuron,n_time);
xT = zeros(J,n_time);
xE = xT;

for i = 1:n_time-1
    xT(:,i+1) = (eye(J)+A*dt)*xT(:,i) +dt*c(:,i);
    V(:,i+1) = (1-lambdaV*dt)*V(:,i)...
        + dt*F'*c(:,i)...
        + dt*Ws*rate(:,i)...
        + 0*Wf*spikes(:,i)...
        + dt*0*randn(n_neuron,1);


    [val,idx] = max(V(:,i+1)-Threshold);

    cnt = 0;
    while val >0
        spikes(idx,i+1) = spikes(idx,i+1) + 1;
        V(:,i+1) = V(:,i+1) + Wf(:,idx);


        [val,idx] = max(V(:,i+1)-Threshold);
        cnt = cnt + 1;


        if cnt> 5e3
        error("Network didnt converge properly");
        end
    end

    if val > 0
        spikes(idx,i+1) = 1;
    end

    rate(:,i+1) = (1-lambdaD*dt)*rate(:,i) + spikes(:,i+1);
    xE(:,i+1) = F*rate(:,i+1);
    
end
end
