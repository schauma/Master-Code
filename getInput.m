function [A,c,x] = getInput(t,lambdaS)

M= length(t);

jumpsX = [0,0,600,600,600,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
sigma = 0.0;
cx = createStep(jumpsX, M,sigma);
cy = createStep(-jumpsX*0.75, M,sigma);

A = -lambdaS;
x0 = 0;
c = cx;
% 
% c= 100*sin(2*pi*t);
% 
%c = [1*sin(2*pi*t);1*cos(2*pi*t)];




% System
% dx = Ax + c(t)
% x0 =  [0;0]; % Initial state
% A = [-4.8,-22.4;40,0];
% c = [cx;0*cy];
% 
x0 =  [0;0]; % Initial state
A = [-400,-800;50,0];
c = [cx;0*cy];


% 
% 
% A = [0,0,1,0;
%     0,0,0,1;
%     0,0,-lambdaS, 0;
%     0,0,0,-lambdaS];
% x0 = [0.2;0.2;0;0];
% c = [cx*0;cx*0;cx;cy];
% 


 J = size(A,1); % Number dynamic variables
 x = zeros(J,M);
 x(:,1) = x0;
 c = smoothdata(c,2,"movmean",floor(1/200*M));
end


function c = createStep(jumps, M,sigma)
c = reshape(repmat(jumps,[ceil(M/length(jumps)),1]),[],1);
c = c(1:M)'; %Some stepwise input
c = c + sigma.*randn(size(c));

end
