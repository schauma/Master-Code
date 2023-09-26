clc
Ws
tmp = round(Ws - F*A*F',9)
FF = F*F'

frac = tmp./(lambdaD*FF);


frac(abs(frac)>1e12) = 0
