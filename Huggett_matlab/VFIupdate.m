function [V1,policy_ind] = VFIupdate(V0,P,U,beta,asize,ssize)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
 

        EV = (V0*P')'; % (sszie*asize) Mx
        %     EV(a1'|z1)...EV(an'|z1)
        % EV= EV(a1'|z2)
        %        :
        %     EV(a1'|zm)...EV(an'|zm)
        
        W = U + beta*kron(EV,ones(asize,1)); % total return function (asize*ssize)*(asize) matrix
        
        [V1,policy_ind]= max(W,[],2);        % V1 => (aszie*ssize)*1
        
        V1 = reshape(V1,asize,ssize); % V1 => sszie*asize
end

