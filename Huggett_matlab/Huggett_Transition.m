figure(1)
clf

maxit=3;
T=100;

xi=0.04*exp(-[0:maxit-1]*0.2)';

r0t=r*ones(T,1);

V=zeros(asize,ssize,T);

Vt(:,:,T)=V0;

distrt=zeros(asize,ssize,T);

distrt(:,:,1)=distr;

wt=ones(T,1) +0.1*exp(-0.1*[0:T-1]');

At=zeros(T,1);
At(1)=B;

for it=1:maxit

        
%backward iteration

for i=1:T-1
   t=T-i;

   tau(t)=r0t(t)*B;
   
    C = max(repmat((1+r0t(t))*grid_a',ssize,asize) + kron(wt(t)*grid_s' -tau(t),ones(asize,asize)) - repmat(grid_a,asize*ssize,1),1e-24);
    
    U = C.^(1-gamma)/(1-gamma); % U-fn
    
    [V(:,:,t),policy_ind(:,t)] = VFIupdate(V(:,:,T),P,U,beta,asize,ssize);
   
end

%forward iteration

for i=1:T-1
    
   t=i;
   
   distl=reshape(distrt(:,:,t),asize*ssize,1);
   
   [H] = DISTupdate(policy_ind(:,t),P,asize,ssize);
    
   distrt(:,:,t+1)=reshape((distl'*H)',asize,ssize);
    
   At(t+1)=sum(sum(distrt(:,:,t+1).*grid_a'));

end


%r0t=r0t+xi(it).*sign(B-A');
figure(1)
hold on
plot(r0t)

end

