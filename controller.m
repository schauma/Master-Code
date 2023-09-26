function [xE,V,rate,spikes,error,forward_signal,rDec,iDec,failed] = controller(A,B,C,ref,dref,Nneuron,dt,M,rDec,iDec,mu,nu,lambdaD,sigmaV,lambdaV)
% A system matrix
% B input matrix
% C output matrix
% xT Target
% Nneuron number neurons
% dt timestep
% M number of timesteps
% iDec instantanious Decoder
% rDec rate Decoder
% mu regularization L1
% nu regularization L2
% lambdaD decoder leak
% sigmaV noise
% lambdaV voltage leak  
% ALLOW_LEARNING Activates the learning of connectivities


% Return
% xE Target Estimation
% V voltage of neurons over time
% rate filtered firing rate
% spikes Individual spike trains
% forward_signal feedforward signal
% rDec the learned FeedForward weights
% iDec the learned Recurrent weights

failed = false;
spikes = zeros(Nneuron,M);
V = zeros(Nneuron,M);
rate = zeros(Nneuron,M);
J = size(A,1); % Number of state variables
b = size(B,2); % Number of input variables
l = size(C,1); % Number of output variables
xE= zeros(J,M); % Network estimate
u = zeros(b,M); % control signal
forward_signal = zeros(J,M); % Feedforward signal
error = zeros(J,M); % tracking error


Thresh= (diag(iDec'*(B'*(C'*C)*B)*iDec) + nu*lambdaD + mu*lambdaD^2)/2;

Ws = -iDec'*(B'*(C'*C)*B)*rDec/lambdaD + 1*mu*lambdaD^2.*eye(Nneuron);
Wf = -iDec'*(B'*(C'*C)*B)*iDec - mu*lambdaD^2.*eye(Nneuron);


for i = 1:M-1

    noise = randn(Nneuron,1);
    e = ref(:,i) - xE(:,i);
    e = -1*e; % Error from paper!
    error(:,i) = e; % Errorl of timestep
    V(:,i+1) = (1-lambdaV*dt)*V(:,i)...
        + dt*iDec'*B'*C'*C*A*error(:,i)...
        + 1*dt*iDec'*B'*C'*C*forward_signal(:,i)...
        + dt*Ws*rate(:,i)...
        + 0*Wf*spikes(:,i)...M
        + sqrt(dt)*sigmaV*noise;

    [m,k] = max(V(:,i+1)-Thresh);
    s = 0;
    while m>0
        spikes(k,i+1) = spikes(k,i+1) + 1;
        V(:,i+1) = V(:,i+1) + Wf(:,k);
        [m,k] = max(V(:,i+1)-Thresh);
        s = s+1;
        if s > 1
            break
        end
        if (s>5e6)
            failed = true;
            return
        end
    end

    if m>0 && 0 %Deactivate the if
        spikes(k,i+1) = 1;
    end
    % WRONG! 
%     ff = V(:,i+1)>Thresh;
%     spikes(ff,i+1) = 1;

    rate(:,i+1) = (1-dt*lambdaD)*rate(:,i) + lambdaD*spikes(:,i+1);
    
    u(:,i+1) = rDec/lambdaD*rate(:,i+1) + iDec/dt*spikes(:,i+1);
   
    xE(:,i+1) = (eye(J) + dt*A)*xE(:,i) + dt*B*u(:,i+1);

    % maybe explicitly compute the derivative of xE with the state
    % equation??
    % the multiply by 2 is fixing it?
    % yes, for the step at least
    forward_signal(:,i+1) =  (dref(:,i+1)- A*xE(:,i+1));

end
end
