function [f_tan,f_norm]=stance_force_three_link(x,dx,u)
% STANCE_FORCE_THREE_LINK    Calculate the forces on the stance
%                            leg during impact.
%    [F_TAN,F_NORM] = STANCE_FORCE_THREE_LINK(X,DX,U) are the forces on the
%    stance leg at impact.

% Eric Westervelt
% 20-Feb-2025 07:33:11

[r,m,Mh,Mt,L,g]=model_params_three_link;

[th3d,th1d,alpha,epsilon]=control_params_three_link;

th1=x(1); th2=x(2); th3=x(3);
dth1=x(4); dth2=x(5); dth3=x(6);

% De11 matrix
De11=zeros(3,3);
De11(1,1)=(r^2*(4*Mh + 4*Mt + 5*m))/4;
De11(1,2)=-(m*r^2*(cos(th1)*cos(th2) + sin(th1)*sin(th2)))/2;
De11(1,3)=L*Mt*r*(cos(th1)*cos(th3) + sin(th1)*sin(th3));
De11(2,1)=-(m*r^2*(cos(th1)*cos(th2) + sin(th1)*sin(th2)))/2;
De11(2,2)=(m*r^2)/4;
De11(3,1)=L*Mt*r*(cos(th1)*cos(th3) + sin(th1)*sin(th3));
De11(3,3)=L^2*Mt;

% De12 matrix
De12=zeros(3,2);
De12(1,1)=(r*cos(th1)*(2*Mh + 2*Mt + 3*m))/2;
De12(1,2)=-(r*sin(th1)*(2*Mh + 2*Mt + 3*m))/2;
De12(2,1)=-(m*r*cos(th2))/2;
De12(2,2)=(m*r*sin(th2))/2;
De12(3,1)=L*Mt*cos(th3);
De12(3,2)=-L*Mt*sin(th3);

% De22 matrix
De22=zeros(2,2);
De22(1,1)=Mh + Mt + 2*m;
De22(2,2)=Mh + Mt + 2*m;

% Ce11 matrix
Ce11=zeros(3,3);
Ce11(1,2)=(dth2*m*r^2*(cos(th1)*sin(th2) - cos(th2)*sin(th1)))/2;
Ce11(1,3)=-L*Mt*dth3*r*(cos(th1)*sin(th3) - cos(th3)*sin(th1));
Ce11(2,1)=-(dth1*m*r^2*(cos(th1)*sin(th2) - cos(th2)*sin(th1)))/2;
Ce11(3,1)=L*Mt*dth1*r*(cos(th1)*sin(th3) - cos(th3)*sin(th1));

% Ce21 matrix
Ce21=zeros(2,3);
Ce21(1,1)=-(dth1*r*sin(th1)*(2*Mh + 2*Mt + 3*m))/2;
Ce21(1,2)=(dth2*m*r*sin(th2))/2;
Ce21(1,3)=-L*Mt*dth3*sin(th3);
Ce21(2,1)=-(dth1*r*cos(th1)*(2*Mh + 2*Mt + 3*m))/2;
Ce21(2,2)=(dth2*m*r*cos(th2))/2;
Ce21(2,3)=-L*Mt*dth3*cos(th3);

% Ge1 matrix
Ge1=zeros(3,1);
Ge1(1,1)=- Mh*g*r*sin(th1) - Mt*g*r*sin(th1) - (3*g*m*r*sin(th1))/2;
Ge1(2,1)=(g*m*r*sin(th2))/2;
Ge1(3,1)=-L*Mt*g*sin(th3);

% Ge2 matrix
Ge2=zeros(2,1);
Ge2(2,1)=Mh*g + Mt*g + 2*g*m;

% B matrix
B=zeros(3,2);
B(1,1)=-1;
B(2,2)=-1;
B(3,1)=1;
B(3,2)=1;

% See my notes, 2/16/200 for equations...
DD=inv((De12*inv(De22)).'*De12*inv(De22))*(De12*inv(De22)).';
F=DD*(-(De11-De12*inv(De22)*De12.')...
  *dx(4:6)+(De12*inv(De22)*Ce21-Ce11)...
  *dx(1:3)+De12*inv(De22)*Ge2-Ge1+B*u);

f_tan=F(1);
f_norm=F(2);
