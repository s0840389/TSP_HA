%==========================================================================
%
% Simulating Markov chain and estimate AR(1) 
%
%==========================================================================
clc
clear
close all

%% Q1. Simulate markov chain and estimate persistence and LR variance

% Generate Grids

grid.s = [0.6177 0.8327 1.0000 1.2009 1.6188];  % labour productivity

P  = [0.7497  0.2161 0.0322 0.002  0      ; 
      0.2161  0.4708 0.2569 0.0542 0.002  ; 
      0.0322  0.2569 0.4218 0.2569 0.0322 ; 
      0.002   0.0542 0.2569 0.4708 0.2161 ; 
      0       0.002  0.0322 0.2161 0.7497 ];    % Markov Transition Mx
P = P'; % Does not need in this case since P is symmetric 
  
% Simulation

discard = 1000; % discard initial draws to eliminate the initial effect
t = 50000 + discard; % number of simulation

zz = rand(t,1);  % vector of random numbers (simulation)
ss = zeros(t,1); % storage for state
ss(1) = 1;

PC = cumsum(P,2); % Cumulative probability of Transition 
% cumulative sum of rows, we will use this to decide on the next step.
% Below is what cumsum does
% for j=1:n-1
%     PC(:,j+1) = PC(:,j) + P(:,j+1);
% end

% Actual simulation
% step by step approach
for i=2:t
    if zz(i) <= PC(ss(i-1),1)
       ss(i) = 1;
    else if zz(i) <= PC(ss(i-1),2)
            ss(i) = 2;
        else if zz(i) <= PC(ss(i-1),3)
                ss(i) = 3;
            else if zz(i) <= PC(ss(i-1),4)
                    ss(i) = 4;
                 else if zz(i) <= PC(ss(i-1),5)
                    ss(i) = 5;
                     end
                end
            end
        end
    end
end
s = grid.s(ss)';

% One step approach from Dr. Ralph
for i=2:t
ss(i) = min(length(grid.s), sum(PC(ss(i-1),:)< zz(i))+1);
end
s = grid.s(ss)';

% One step approach thanks to Matthew Read
% for i = 2:t
% ss(i) = grid.s(find(rand < PC(grid.s == ss(i-1),:),1,'first'));
% end

% Use OLS to estimate persistence and variance
% One could also estimate OLS without constant
s_t1 = s(discard+1:end);
s_t  = s(discard:end-1);
s_tv = [ones(length(s_t),1), s_t]; 
betav = regress(s_t1,s_tv);

beta = betav(2);
persistence = betav(2);             % AR(1) parameter
resid = s_t1 - s_t*beta;            % Find residual
SRvar = var(resid);                 % SR var
LRvar = SRvar/(1-persistence^2);    % LR var
LRvar_check = var(s(discard:end));  % check that we get the same solution
disp('persistence is :        '); disp(persistence);
disp('LR variance is : ');disp(LRvar);

%% Q2. Find stationary distribution over S and determine the total labor supply

% Use eigenvector to find stationary distribution
[xi, eigen] = eigs(P',1); 
% invariant dist'n is left e-vector of transition mx associated to lambda=1
xi = xi./sum(xi); % Have to normalize the e-vector. 
% Def of ergodic dist'n : xi(t+1)' = xi(t)'*PI => PI'*xi(t) = xi(t+1)
N = grid.s*xi; % Total labor supply
