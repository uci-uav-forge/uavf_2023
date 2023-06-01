% Notes on deriving the state space model for quadrotors


% phi: roll, theta: pitch, psi: yaw
syms phi theta psi phi_dot theta_dot psi_dot;
ang = [phi; theta; psi];
ang_d = [phi_dot; theta_dot; psi_dot];



% Rotation matrix R_ZYX from body frame to inertial frame
% NECESSARY FOR LINEAR ACCELERATIONS IN STATE-DOT
Rx = [ 1,           0,          0;
       0,           cos(phi),  -sin(phi);
       0,           sin(phi),   cos(phi) ];
Ry = [ cos(theta),  0,          sin(theta);
       0,           1,          0;
      -sin(theta),  0,          cos(theta) ];
Rz = [cos(psi),    -sin(psi),   0;
      sin(psi),     cos(psi),   0;
      0,            0,          1 ];
% rotation matrix from body frame to inertial frame
R = Rz*Ry*Rx; 

disp('Rotation matrix:')
latex(R)



% Transformation matrix for angular velocities from inertial to body frame
% Necessary for the Jacobian that converts body to inertial frame velocites
W = [ 1,  0,        -sin(theta);
      0,  cos(phi),  cos(theta)*sin(phi);   
      0, -sin(phi),  cos(theta)*cos(phi) ];
syms Ixx Iyy Izz
I = [Ixx, 0, 0; 0, Iyy, 0; 0, 0, Izz];
% Jacobian that converts body frame velocities to inertial frame
J = W.'*I*W;

disp('Body to inertial frame velocity conversion:')
latex(J)



% Coriolis matrix for defining equations of motion
% NECESSARY FOR ANGULAR ACCELERATIONS IN STATE-DOT
C11 = 0;

C12 = (Iyy-Izz)*(theta_dot*cos(phi)*sin(phi) + psi_dot*(sin(phi)^2)*cos(theta)) +...
    (Izz-Iyy)*psi_dot*(cos(phi)^2)*cos(theta) -...
    Ixx*psi_dot*cos(theta);

C13 = (Izz-Iyy)*psi_dot*cos(phi)*sin(phi)*(cos(theta)^2);

C21 = (Izz-Iyy)*(theta_dot*cos(phi)*sin(phi) + psi_dot*(sin(phi)^2)*cos(theta)) +...
    (Iyy-Izz)*psi_dot*(cos(phi)^2)*cos(theta) +...
    Ixx*psi_dot*cos(theta);

C22 = (Izz-Iyy)*phi_dot*cos(phi)*sin(phi);

C23 = -Ixx*psi_dot*sin(theta)*cos(theta) +...
    Iyy*psi_dot*(sin(phi)^2)*sin(theta)*cos(theta) +...
    Izz*psi_dot*(cos(phi)^2)*sin(theta)*cos(theta);

C31 = (Iyy-Izz)*psi_dot*(cos(theta)^2)*sin(phi)*cos(phi) -...
    Ixx*theta_dot*cos(theta);

C32 = (Izz-Iyy)*(theta_dot*cos(phi)*sin(phi)*sin(theta) + phi_dot*(sin(phi)^2)*cos(theta)) +...
    (Iyy-Izz)*phi_dot*(cos(phi)^2)*cos(theta) +...
    Ixx*psi_dot*sin(theta)*cos(theta) -...
    Iyy*psi_dot*(sin(phi)^2)*sin(theta)*cos(theta) -...
    Izz*psi_dot*(cos(phi)^2)*sin(theta)*cos(theta);

C33 = (Iyy-Izz)*phi_dot*cos(phi)*sin(phi)*(cos(theta)^2) -...
    Iyy*theta_dot*(sin(phi)^2)*cos(theta)*sin(theta) -...
    Izz*theta_dot*(cos(phi)^2)*cos(theta)*sin(theta) +...
    Ixx*theta_dot*cos(theta)*sin(theta);

C = [C11 C12 C13; C21 C22 C23; C31 C32 C33];
C_lin = subs(C, ...
    {phi,theta,phi_dot,theta_dot,psi_dot}, ...
    {0,0,0,0,0} ...
);

disp('Coriolis matrix:')
latex(C)
latex(C_lin)




% k: lift constant
% m: total mass
% b: drag constant
% l: dist between rotor and COM
% D: coeff describing overall drag due to velocity
syms k m l b g Dx Dy Dz
D = [Dx 0 0; 0 Dy 0; 0 0 Dz];
% square of motor frequency
syms u1 u2 u3 u4
u = [u1; u2; u3; u4];



% positional vars
syms x y z x_dot y_dot z_dot;
pos = [x; y; z]; 
pos_d = [x_dot; y_dot; z_dot];

% positional eqs of motion for system matrix
pos_dd_A = [0;0;g] - (1/m)*D*pos_d;
pos_dd_B = R*(k/m)*[0 0 0 0; 0 0 0 0; 1 1 1 1]*u;

% angular eqs of motion
ang_dd_A = -inv(J)*C*ang_d;
ang_dd_B = inv(J)*[0 -l*k 0 l*k; -l*k 0 l*k 0; -b b -b b]*u;

disp('Positional equations of motion:')
latex(pos_dd_A + pos_dd_B)
disp('Angular equations of motion:')
latex(ang_dd_A + ang_dd_B)



% continuous-time system matrix linearized around hover
A = jacobian( ...
    [pos_d; ang_d; pos_dd_A; ang_dd_A], ...
    [pos; ang; pos_d; ang_d] ...
);
B = jacobian( ...
    [pos_d; ang_d; pos_dd_B; ang_dd_B], ...
    u ...
);

A_lin = subs(A, ...
    {phi,theta,phi_dot,theta_dot,psi_dot}, ...
    {0,0,0,0,0} ...
);
B_lin = subs(B, ...
    {phi,theta,phi_dot,theta_dot,psi_dot}, ...
    {0,0,0,0,0} ...
);

disp('Nonlinear system and input matrices:')
latex(A)
latex(B)
disp('Linearized system and input matrices:')
latex(A_lin)
latex(B_lin)



% converting continuous-time state space to discrete-time
syms T

% exact discretization of linearized A
Ad_exact_lin = expm(A_lin*T);

% backwards euler method approximation
Ad_euler = (eye(size(A)) + 0.5*A*T) * inv(eye(size(A)) - 0.5*A*T);
%Bd = inv(A) * (Ad - eye(size(Ad))) * B;
Ad_euler_lin = subs(Ad_euler, ...
    {phi,theta,phi_dot,theta_dot,psi_dot}, ...
    {0,0,0,0,0} ...
);

%{
Bd_lin = subs(Bd, ...
    {phi,theta,phi_dot,theta_dot,psi_dot}, ...
    {0,0,0,0,0} ...
);
%}

disp('Lineared system matrix discretized with matrix exponential')
latex(Ad_exact_lin)
disp('Linearized system matrix discretized with backwards euler method')
latex(Ad_euler_lin)
%latex(Bd_lin)

%save('state_space.mat', 'A', 'B', 'A_lin', 'B_lin', 'Ad', 'Ad_lin');



%{
Ad_lin = zeros(size(A));
Bd_lin = zeros(size(B));
for k=1:400
    Ad_lin = Ad_lin + (1/factorial(k))*(A_lin*T)^k;
    Bd_lin = Bd_lin + (1/factorial(k))*(A_lin^(k-1))*(T^k)*B_lin;
end
%}

%Bd = inv(Ac)*(Ad - eye(3,3))*Bc
%{
Bd = zeros(3,2);
for k=1:100
    Bd = Bd + (1/factorial(k))*(Ac^(k-1))*(T^k)*Bc;
end
%}


