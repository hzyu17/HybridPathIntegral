function [x_new,z2_new]=transition_three_link(x)
% TRANSITION_THREE_LINK    Calculate the state of the system after impact.
%          (Last two entries are the forces at the toe.
%    [X_NEW,Z2_NEW] = TRANSITION_THREE_LINK(X) is the transition function for
%    biped walking model. (x is of dimension 8)

% Eric Westervelt
% 20-Feb-2025 07:33:11

[r,m,Mh,Mt,L,g]=model_params_three_link;

th1=x(1); th2=x(2); th3=x(3);

% De matrix
De=zeros(5,5);
De(1,1)=(r^2*(4*Mh + 4*Mt + 5*m))/4;
De(1,2)=-(m*r^2*(cos(th1)*cos(th2) + sin(th1)*sin(th2)))/2;
De(1,3)=L*Mt*r*(cos(th1)*cos(th3) + sin(th1)*sin(th3));
De(1,4)=(r*cos(th1)*(2*Mh + 2*Mt + 3*m))/2;
De(1,5)=-(r*sin(th1)*(2*Mh + 2*Mt + 3*m))/2;
De(2,1)=-(m*r^2*(cos(th1)*cos(th2) + sin(th1)*sin(th2)))/2;
De(2,2)=(m*r^2)/4;
De(2,4)=-(m*r*cos(th2))/2;
De(2,5)=(m*r*sin(th2))/2;
De(3,1)=L*Mt*r*(cos(th1)*cos(th3) + sin(th1)*sin(th3));
De(3,3)=L^2*Mt;
De(3,4)=L*Mt*cos(th3);
De(3,5)=-L*Mt*sin(th3);
De(4,1)=(r*cos(th1)*(2*Mh + 2*Mt + 3*m))/2;
De(4,2)=-(m*r*cos(th2))/2;
De(4,3)=L*Mt*cos(th3);
De(4,4)=Mh + Mt + 2*m;
De(5,1)=-(r*sin(th1)*(2*Mh + 2*Mt + 3*m))/2;
De(5,2)=(m*r*sin(th2))/2;
De(5,3)=-L*Mt*sin(th3);
De(5,5)=Mh + Mt + 2*m;

% E matrix
E=zeros(2,5);
E(1,1)=r*cos(th1);
E(1,2)=-r*cos(th2);
E(1,4)=1;
E(2,1)=-r*sin(th1);
E(2,2)=r*sin(th2);
E(2,5)=1;

% See Grizzle's paper, page 28 for equation...
tmp_vec=inv([De -E';E zeros(2)])*[De*[x(4:6)';zeros(2,1)];zeros(2,1)];

x_new(1)=x(2);
x_new(2)=x(1);
x_new(3)=x(3);
x_new(4)=tmp_vec(2);
x_new(5)=tmp_vec(1);
x_new(6)=tmp_vec(3);
x_new(7)=tmp_vec(6);
x_new(8)=tmp_vec(7);
z2_new=tmp_vec(5);
