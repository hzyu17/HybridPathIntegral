% A MATLAB script to generate the equations of motion for a 5-link,
% planar, kneed, biped walker using the method of Lagrange.
%
% This routine is for a robot with an upright torso, two legs and
% knees.  The model is for five degrees of freedom of the robot,
% with the stance foot (Foot1) pinned to the ground.  The notation
% used for the equation of motions are as in Robot Dynamics and
% Control by Spong and Vidyasagar (1989), page 142, Eq. (6.3.12)
%
%    D(q)ddq + C(q,dq)*dq + G(q) = B*tau + E2*F_external
%
% Note the following convention
%
%  fem = femur, tib = tibia, 1 = stance leg, 2 = swing leg
%
% For simplicity, the model is derived using absolute coorinates
% measured in the trigonometric sense from the vertical.  The model
% is trasnformed in to relative coordates plus one absolute
% coordinates via a coordinate transformation.
%
% This file is associated with the book Feedback Control of Dynamic 
% Bipedal Robot Locomotion by Eric R. Westervelt, Jessy W. Grizzle, 
% Christine Chevallereau, Jun-Ho Choi, and Benjamin Morris published 
% by Taylor & Francis/CRC Press in 2007.
% 
% Copyright (c) 2007 by Eric R. Westervelt, Jessy W. Grizzle, Christine
% Chevallereau, Jun-Ho Choi, and Benjamin Morris.  This code may be
% freely used for noncommercial ends.  If use of this code in part or in
% whole results in publication, proper citation must be included in that
% publication.  This code comes with no guarantees or support.
% 
% Eric Westervelt
% 20 February 2007

clear all

% -----------------------------------------------------------------
%
%  Model variables
%
% -----------------------------------------------------------------

% absolute joint angles and velocities
syms q_torso dq_torso real
syms q_fem1 dq_fem1 real
syms q_fem2 dq_fem2 real
syms q_tib1 dq_tib1 real
syms q_tib2 dq_tib2 real

% gravity
syms g real

% link lengths
syms L_torso real
syms L_fem real
syms L_tib real

% link masses
syms M_torso real
syms M_fem real
syms M_tib real

% link inertias
syms MY_torso real
syms MZ_torso real
syms MZ_fem real
syms MZ_tib real

% center of mass offsets
syms XX_torso real
syms XX_fem real
syms XX_tib real

% relative joint angles
syms q31L q32L q41L q42L q1L real
q  = [q31L; q32L; q41L; q42L; q1L];

% relative joint velocities
syms dq31L dq32L dq41L dq42L dq1L real
dq = [dq31L; dq32L; dq41L; dq42L; dq1L];

% -----------------------------------------------------------------
%
%  Change coordinates to relative coordinates
%
% -----------------------------------------------------------------

T = [ 1   0   0   0   1;
	0   1   0   0   1;
	1   0   1   0   1;
	0   1   0   1   1;
	0   0   0   0   1];

% old absolute coordinates in terms of relative coords
q_new  = inv(T)*q;
dq_new = inv(T)*dq;

q_fem1  = q_new(1);
q_fem2  = q_new(2);
q_tib1  = q_new(3);
q_tib2  = q_new(4);
q_torso = q_new(5);
dq_fem1  = dq_new(1);
dq_fem2  = dq_new(2);
dq_tib1  = dq_new(3);
dq_tib2  = dq_new(4);
dq_torso = dq_new(5);

%% Generate De matrix.  To do so, we must go through the trouble of
%% defining the positions of the masses using the augmented
%% configuration variables.

syms z1 z2 dz1 dz2 real

%% generalized coordinates
qe=[q31L;q32L;q41L;q42L;q1L;z1;z2];

%% first derivative of generalized coordinates
dqe=[dq31L;dq32L;dq41L;dq42L;dq1L;dz1;dz2];

%%

% -----------------------------------------------------------------
%
%  Calculate kinetic energy
%
% -----------------------------------------------------------------

% hip and knee positions
p_hip  = L_fem*[sin(q_fem1); -cos(q_fem1)] ...
	+ L_tib*[sin(q_tib1); -cos(q_tib1)] + [z1; z2];
p_knee1 = p_hip + L_fem*[-sin(q_fem1); cos(q_fem1)];
p_knee2 = p_hip + L_fem*[-sin(q_fem2); cos(q_fem2)];

%%
% hip and knee velocities
v_hip  = jacobian(p_hip,qe)*dqe;
v_knee1 = jacobian(p_knee1,qe)*dqe;
v_knee2 = jacobian(p_knee2,qe)*dqe;

% relative angular velocties -- needed since the centers of masses
% of the links are one collocated with the link reference frames
R_torso = [cos(q_torso) -sin(q_torso);
	sin(q_torso) cos(q_torso)];
v_torso = R_torso.'*v_hip*dq_torso;

R_fem1 = [cos(q_fem1) -sin(q_fem1);
	sin(q_fem1) cos(q_fem1)];
v_fem1 = R_fem1.'*v_hip*dq_fem1;

R_fem2 = [cos(q_fem2) -sin(q_fem2);
	sin(q_fem2) cos(q_fem2)];
v_fem2 = R_fem2.'*v_hip*dq_fem2;

R_tib1 = [cos(q_tib1) -sin(q_tib1);
	sin(q_tib1) cos(q_tib1)];
v_tib1 = R_tib1.'*v_knee1*dq_tib1;

R_tib2 = [cos(q_tib2) -sin(q_tib2);
	sin(q_tib2) cos(q_tib2)];
v_tib2 = R_tib2.'*v_knee2*dq_tib2;

% kinetic energy of links
KE_torso = 1/2*M_torso*v_hip.'*v_hip ...
	+ v_torso.'*[-MZ_torso; MY_torso]  ...
	+ 1/2*XX_torso*dq_torso^2;
KE_torso = simplify(KE_torso);
KE_fem1 = 1/2*M_fem*v_hip.'*v_hip ...
	+ v_fem1.'*[-MZ_fem; 0] ...
	+ 1/2*XX_fem*(dq_fem1)^2;
KE_fem1    = simplify(KE_fem1);
KE_fem2 = 1/2*M_fem*v_hip.'*v_hip ...
	+ v_fem2.'*[-MZ_fem; 0] ...
	+ 1/2*XX_fem*(dq_fem2)^2;
KE_fem2    = simplify(KE_fem2);
KE_tib1 = 1/2*M_tib*v_knee1.'*v_knee1 ...
	+ v_tib1.'*[-MZ_tib; 0] ...
	+ 1/2*XX_tib*(dq_tib1)^2;
KE_tib1 = simplify(KE_tib1);
KE_tib2 = 1/2*M_tib*v_knee2.'*v_knee2 ...
	+ v_tib2.'*[-MZ_tib;0] ...
	+ 1/2*XX_tib*(dq_tib2)^2;
KE_tib2 = simplify(KE_tib2);

% total kinetic energy
KE = KE_torso + KE_fem1 + KE_fem2 + KE_tib1 + KE_tib2;
KE = simplify(KE);
%%

De_mtx=simplify(jacobian(jacobian(KE,dqe).',dqe));


%% This part computes the matrices and vectors 
%% of the pinned dynamics, D,C,B,G

% -----------------------------------------------------------------
%
%  Calculate kinetic energy
%
% -----------------------------------------------------------------

% hip and knee positions
p_hip  = L_fem*[sin(q_fem1); -cos(q_fem1)] ...
	+ L_tib*[sin(q_tib1); -cos(q_tib1)];
p_knee1 = p_hip + L_fem*[-sin(q_fem1); cos(q_fem1)];
p_knee2 = p_hip + L_fem*[-sin(q_fem2); cos(q_fem2)];

% hip and knee velocities
v_hip  = jacobian(p_hip,q)*dq;
v_knee1 = jacobian(p_knee1,q)*dq;
v_knee2 = jacobian(p_knee2,q)*dq;

% relative angular velocties -- needed since the centers of masses
% of the links are one collocated with the link reference frames
R_torso = [cos(q_torso) -sin(q_torso);
	sin(q_torso) cos(q_torso)];
v_torso = R_torso.'*v_hip*dq_torso;

R_fem1 = [cos(q_fem1) -sin(q_fem1);
	sin(q_fem1) cos(q_fem1)];
v_fem1 = R_fem1.'*v_hip*dq_fem1;

R_fem2 = [cos(q_fem2) -sin(q_fem2);
	sin(q_fem2) cos(q_fem2)];
v_fem2 = R_fem2.'*v_hip*dq_fem2;

R_tib1 = [cos(q_tib1) -sin(q_tib1);
	sin(q_tib1) cos(q_tib1)];
v_tib1 = R_tib1.'*v_knee1*dq_tib1;

R_tib2 = [cos(q_tib2) -sin(q_tib2);
	sin(q_tib2) cos(q_tib2)];
v_tib2 = R_tib2.'*v_knee2*dq_tib2;

% kinetic energy of links
KE_torso = 1/2*M_torso*v_hip.'*v_hip ...
	+ v_torso.'*[-MZ_torso; MY_torso]  ...
	+ 1/2*XX_torso*dq_torso^2;
KE_torso = simplify(KE_torso);
KE_fem1 = 1/2*M_fem*v_hip.'*v_hip ...
	+ v_fem1.'*[-MZ_fem; 0] ...
	+ 1/2*XX_fem*(dq_fem1)^2;
KE_fem1    = simplify(KE_fem1);
KE_fem2 = 1/2*M_fem*v_hip.'*v_hip ...
	+ v_fem2.'*[-MZ_fem; 0] ...
	+ 1/2*XX_fem*(dq_fem2)^2;
KE_fem2    = simplify(KE_fem2);
KE_tib1 = 1/2*M_tib*v_knee1.'*v_knee1 ...
	+ v_tib1.'*[-MZ_tib; 0] ...
	+ 1/2*XX_tib*(dq_tib1)^2;
KE_tib1 = simplify(KE_tib1);
KE_tib2 = 1/2*M_tib*v_knee2.'*v_knee2 ...
	+ v_tib2.'*[-MZ_tib;0] ...
	+ 1/2*XX_tib*(dq_tib2)^2;
KE_tib2 = simplify(KE_tib2);

% total kinetic energy
KE = KE_torso + KE_fem1 + KE_fem2 + KE_tib1 + KE_tib2;
KE = simplify(KE);

% -----------------------------------------------------------------
%
%  Calculate potential energy
%
% -----------------------------------------------------------------

% positions of various members
p_torso = p_hip ...
	+ 1/M_torso*[-sin(q_torso)*MZ_torso - cos(q_torso)*MY_torso;
	cos(q_torso)*MZ_torso + sin(q_torso)*MY_torso];

p_fem1  = p_hip + MZ_fem/M_fem*[-sin(q_fem1); cos(q_fem1)];
p_fem2  = p_hip + MZ_fem/M_fem*[-sin(q_fem2); cos(q_fem2)];
p_tib1  = p_knee1 + MZ_tib/M_tib*[-sin(q_tib1); cos(q_tib1)];
p_tib2  = p_knee2 + MZ_tib/M_tib*[-sin(q_tib2); cos(q_tib2)];
p_foot1 = p_knee1 + L_tib*[-sin(q_tib1); cos(q_tib1)];
p_foot2 = p_knee2 + L_tib*[-sin(q_tib2); cos(q_tib2)];

% total potential energy
PE = g*(M_torso*p_torso(2) + M_fem*p_fem1(2) + M_fem*p_fem2(2) + ...
	M_tib*p_tib1(2) + M_tib*p_tib2(2));
PE = simplify(PE);

% -----------------------------------------------------------------
%
%  Calculate model matrices
%
% -----------------------------------------------------------------

% gravity vector
G_vect = jacobian(PE,q).';
G_vect = simplify(G_vect);

% mass-inertial matrix
D_mtx = simplify(jacobian(KE,dq).');
D_mtx = simplify(jacobian(D_mtx,dq));

% Coriolis and centrifugal matrix
syms C_mtx real
n=max(size(q));
for k=1:n
	for j=1:n
		C_mtx(k,j)=0*g;
		for i=1:n
			C_mtx(k,j)=C_mtx(k,j)+1/2*(diff(D_mtx(k,j),q(i)) + ...
				diff(D_mtx(k,i),q(j)) - ...
				diff(D_mtx(i,j),q(k)))*dq(i);
		end
	end
end
C_mtx=simplify(C_mtx);

% input matrix
Phi_0 = [q_fem1-q_torso;
	q_fem2-q_torso;
	q_tib1-q_fem1;
	q_tib2-q_fem2;
	q_torso];
B_mtx = jacobian(Phi_0,q);
B_mtx = B_mtx.'*[eye(4,4);zeros(1,4)];

% swing foot force input matrix (F_ext = [F_T;F_N])
Phi_1 = [p_foot2];
E_mtx = jacobian(Phi_1,q).';

% ------------------------------------------------------------------------
% Create MATLAB functions for each EOM entry
param_list = {'g','p(1)';
	'L_torso','p(2)'; 'L_fem','p(3)'; 'L_tib','p(4)';
	'M_torso','p(5)'; 'M_fem','p(6)'; 'M_tib','p(7)';
	'MY_torso','p(8)'; 'MZ_torso','p(9)'; 'MZ_fem','p(10)'; 'MZ_tib','p(11)';
	'XX_torso','p(12)'; 'XX_fem','p(13)'; 'XX_tib','p(14)'};

list_q = {'q31L','q(1)'; 'q32L','q(2)'; 'q41L','q(3)'; 'q42L','q(4)'; 'q1L','q(5)'};

list_dq = {'dq31L','dq(1)'; 'dq32L','dq(2)'; 'dq41L','dq(3)'; 'dq42L','dq(4)'; 'dq1L','dq(5)'};

write_fcn('D_matrix.m',{'q','params'},[list_q; param_list],{D_mtx,'D'});
write_fcn('De_matrix.m',{'q','params'},[list_q; param_list],{De_mtx,'De'});
write_fcn('C_matrix.m',{'q','dq','params'},[list_q; list_dq; param_list],{C_mtx,'C'});
write_fcn('G_vector.m',{'q','params'},[list_q; param_list],{G_vect,'G'});
write_fcn('B_matrix.m',{},[],{B_mtx,'B'});
