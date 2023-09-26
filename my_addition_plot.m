clf
lambda = [1,10,20,100,200,500,700,1000];
lambda = lambda';
measurements = [2.3/1000,0.01,0.02,0.1,0.17,0.33,0.42,0.5];
measurements = measurements';
f = fit(lambda,1-measurements, "poly2")
plot(f,lambda,1-measurements,"*")
%set(gca,"YScale","log","XScale","log")
