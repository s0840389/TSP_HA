function [H] = DISTupdate(policy_ind,P,na,ns)
% Use policy function and stochastic shocks to produce joint transition
% matrix H

idm=repmat(policy_ind(:),[1 ns]);
idh=kron(1:ns,ones(1,na*ns));

index = sub2ind([na ns],idm(:),idh(:));

weight=zeros(na,ns,ns);
for hh=1:ns
    % Dimensions (a,s',s)
    weight(:,:,hh)=ones(na,1)*P(hh,:);
end

weight=permute(weight,[1 3 2]);

rowindex=repmat(1:na*ns,[1 ns]);

H=sparse(rowindex(:),index(:),weight(:),na*ns,na*ns);

end

