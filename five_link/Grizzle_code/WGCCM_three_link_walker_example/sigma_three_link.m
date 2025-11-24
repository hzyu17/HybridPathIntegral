function [x]=sigma_three_link(omega_1_minus,a)
% SIGMA_THREE_LINK    Maps velocity of stance leg just before
%         impact to state of the system just before impact.
%    [X] = SIGMA_THREE_LINK(OMEGA_1_MINUS,A)

% Eric Westervelt
% 20-Feb-2025 07:33:11

[th3d,th1d,alpha,epsilon]=control_params_three_link;

a01=a(1); a11=a(2); a21=a(3); a31=a(4);
a02=a(5); a12=a(6); a22=a(7); a32=a(8);

th1=th1d;
dth1=omega_1_minus;

th3 = a01 + a11*th1 + a21*th1^2 + a31*th1^3;
dth2 = dth1*((th1 - th1d)*(a02 + a12*th1 + a22*th1^2 + a32*th1^3) + (th1 + th1d)*(a02 + a12*th1 + a22*th1^2 + a32*th1^3) + (th1 + th1d)*(th1 - th1d)*(a12 + 2*a22*th1 + 3*a32*th1^2) - 1);
dth3 = dth1*(a11 + 2*a21*th1 + 3*a31*th1^2);

x = [th1,-th1,th3,dth1,dth2,dth3];