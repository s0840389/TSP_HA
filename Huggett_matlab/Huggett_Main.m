%==========================================================================
%
% Solving Huggett (94)
%
%==========================================================================

clc
clear
close all

%% 1. Define parameters

% Economic Parameters
phi   = 0;    % Borrowing limit
gamma = 2;     % Coeffcient of relative risk aversion
beta  = 0.96;  % Discount factor
alpha = 1/3;
B=4;


% Numerical parameters
na   = 200;      % Number of points on the asset grid
ns   = 5;        % Number of points on the income grid
crit = 1e-7;    % Numerical precision for outside loop
tol  = 1e-6;    % Numerical precision for inside loop
rh   = 1/beta-1; % Maximum r
init = 5;        % Initial gap for r
gap = 5;         % Initial gap for VF

%% 2. Generate Grids

grid_s = [0.6177 0.8327 1.0000 1.2009 1.6188]; % grid for labor productivity
amin = phi;       % borrowing const
amax = 30;        % arbitrary. We will verify this later

grid_a = linspace(amin,amax,na); % asset grid

P  = [0.7497  0.2161 0.0322 0.002  0      ;
    0.2161  0.4708 0.2569 0.0542 0.002  ;
    0.0322  0.2569 0.4218 0.2569 0.0322 ;
    0.002   0.0542 0.2569 0.4708 0.2161 ;
    0       0.002  0.0322 0.2161 0.7497 ];    % Transition Mx

P_ergodic=P^10000;
NSS=grid_s*P_ergodic(1,:)';

PC = cumsum(P,2); % Cumulative probability of Transition
ssize = length(grid_s);
asize = length(grid_a);

V0 = zeros(asize,ssize);    % Storage for value function
nn = 1;                     % Number of iteration
r = rh;
rl =0;
tic
%---------loop until market clearing---------------------------------------
while abs(init) > crit
    
    tau=r*B;
    
    w = 1;
    
    C = max(repmat((1+r)*grid_a',ssize,asize) + kron(w*grid_s' -tau,ones(asize,asize)) - repmat(grid_a,asize*ssize,1),1e-24);
    
    % Structure of C (asize*ssize)*(asize) matrix
    %     (a1,s1,a1') (a1,s1,a2') ...(a1,s1,an')
    % C = (a2,s1,a1')
    %          :
    %     (an,s1,a1')
    %     (a1,s2,a1')     ...        (a1,s2,an')
    %          :
    %     (an,sm,a1')     ...        (an,sm,an')
    
    U = C.^(1-gamma)/(1-gamma); % U-fn
    
    %---------Value Function Iteration-------------------------------------
    
    disp('Starting Iteration for V. Difference remaining:                      ');
    while gap > tol
        
        
        [V1,policy_ind] = VFIupdate(V0,P,U,beta,asize,ssize);
        
        % Convergence
        gap = norm(V1-V0, inf);    % check convergence criteria
        fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b %20.9f \n',gap);
        
        V0 = V1;
        
    end
    gap = 5;
    
    %%
    %--------Calculate the invariant distribution --------
    
    [H] = DISTupdate(policy_ind,P,na,ns);
    
    % Calculate the unit-eigenvector of the transition matrix
    [distr,eigen]=eigs(H',1);  % Ergodic Distribution as left unit eigenvector
    distr=distr./sum(distr);
    distr= reshape(distr, na,ns);
    
    A=grid_a(policy_ind)*distr(:);
    
    %% update r by bisection
    
    if A-B > 0
        rh=r;
    else
        rl=r;
    end
    
    disp(r)
    disp(A)
    A(nn) = A;
    
    rv(nn) = r;
    init=abs(rh-rl); % want this to be zero
    nn = nn +1;
    
    disp('Starting Iteration for r. Difference remaining:                      ');
    fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b %20.9f \n',init);
    
    r = (rl+rh)/2;  % update r
    
end
toc



