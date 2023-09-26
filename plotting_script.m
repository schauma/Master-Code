%Plotter script

clf
plot_spikes(t(1:10000),spikes(:,1:10000))
hold on
plot(t(1:10000),xE(:,1:10000)*2.5,"MarkerSize",5,"LineWidth" , 5,"Color","red")
plot(t(1:10000),xT(:,1:10000)*2.5,"MarkerSize",5,"LineWidth" , 3, "Color","blue")
grid on
xlabel("time in s","FontSize",15)
title("Controller with learning approach. Timestep 0.1ms", FontSize=15)
