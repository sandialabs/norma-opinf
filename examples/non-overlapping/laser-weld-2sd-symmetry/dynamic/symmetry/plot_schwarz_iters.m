close all; clear all; 

relax = [0.1:0.1:1];
sDN = dlmread('DN/schwarz_iters.txt');
sDNaitken = dlmread('DN-aitken/N0_1/schwarz_iters.txt');
sDNaitken = reshape(sDNaitken, 2, length(sDNaitken)/2);
sDN = reshape(sDN, 2, length(sDN)/2);

close all;
plot(relax(1:length(sDN)), sDN(1,:));
hold on;
plot(relax(1:length(sDNaitken)), sDNaitken(1,:));
legend('classical', 'aitken, N_0=1');
xlabel('\rho^{(0)}');
ylabel('Mean # Schwarz iters');

figure(); 
plot(relax(1:length(sDN)), sDN(2,:));
hold on;
plot(relax(1:length(sDNaitken)), sDNaitken(2,:));
legend('classical', 'aitken, N_0=1');
xlabel('\rho^{(0)}');
ylabel('Max # Schwarz iters');

