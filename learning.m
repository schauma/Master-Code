function [Gamma,Omega] = learning(Nneuron,Nsys,dt,Gamma,Omega,alpha,beta,dG,dO,mu,lambdaV,Thresh)
nTraining = 1e7;
V = zeros(Nneuron,1);
x = zeros(Nsys,1);
spike = 0;
rate = V;
vec = @(x) unitVec(x,Nneuron);

F = Gamma;
C = Omega;

A = 2000;
sigma = 30;
w=(1/(sigma*sqrt(2*pi)))* exp(-(([1:1000]-500).^2)/(2*sigma.^2));%gaussian smoothing kernel used to smooth the input
w=w/sum(w); % normalization oof the kernel
Input=(mvnrnd(zeros(1,Nsys),eye(Nsys),nTraining))'; %generating a new sequence of input which a gaussion vector
for d=1:Nsys
    Input(d,:)=A*conv(Input(d,:),w,'same'); %smoothing the previously generated white noise with the gaussian window w
end
Input  = 50*[sin(dt*(1:nTraining));-cos(dt*(1:nTraining))];
l =1;
k =1;
for i = 1:nTraining-1
    
    
    if ((i/nTraining)>(l/100))
        fprintf('%d percent of the learning completed\n',l)
        l=l+1;
    end

    V=(1-lambdaV*dt)*V + dt*Gamma'*Input(:,i)+ spike*Omega(:,k)+0.001*randn(Nneuron,1); %the membrane potential is a leaky integration of the feedforward input and the spikes
    x=(1-lambdaV*dt)*x+dt*Input(:,i); %filtered input
         
    [m,k]= max(V - Thresh); %finding the neuron with largest membrane potential
    
    if m>= 0
        spike=1; % the spike ariable is turned to one
        Gamma(:,k)=Gamma(:,k)+dG*(alpha*x-Gamma(:,k)); %updating the feedforward weights
        Omega(:,k)=Omega(:,k) -(dO)*(beta*(V+ mu*rate)+Omega(:,k)+mu*vec(k));%updating the recurrent weights
        rate(k,1)=rate(k,1)+1; %updating the filtered spike train
    
    else
        spike = 0;
    end
    rate= (1-dt*lambdaV)*rate;

end
Ntime  = 1e4;
Input = [sin(dt*(1:Ntime));-cos(dt*(1:Ntime))];
[rO, O, V] = runnet(dt, lambdaV, F ,Input, C,Nneuron,Ntime, Thresh);
[rO2, O2, V2] = runnet(dt, lambdaV, Gamma ,Input, Omega,Nneuron,Ntime, Thresh);

[t,y] = ode45(@(t,x) [sin(t);-cos(t)],[0,Ntime*dt],[0;0]);
plot(dt*(1:Ntime),F*rO);
hold on 
plot(dt*(1:Ntime),Gamma*rO2);
plot(t,y);
hold off
end

function v = unitVec(n,d)
v = zeros(d,1);
v(n) = 1;
end
