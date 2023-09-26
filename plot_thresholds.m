function [] = plot_thresholds(Gamma,Nneuron,color)
%PLOT_THRESHOLDS Summary of this function goes here
%   Detailed explanation goes here
for i = Nneuron:-1:1
vec = Gamma(:,i);
normal = [-vec(2);vec(1)];
A = vec + 0.5*normal;
B = vec - 0.5*normal;
plot([A(1),B(1)],[A(2),B(2)],"LineWidth",4,"Color",color(i,:),...
    "Marker",".","MarkerSize",4)
hold on
end
plot(0,0,"+","LineWidth",2,"MarkerSize",20,"MarkerEdgeColor","k")
gridColor = 0.15*ones(4,1);
plot([0,0],ylim,"Color",gridColor);
plot(xlim,[0,0],"Color",gridColor);
end

