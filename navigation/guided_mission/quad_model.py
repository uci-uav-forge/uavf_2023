# source: An integral predictive / nonlinear H ∞ control structure for a quadrotor helicopter - G. Raffo, M. Ortega, F. Rubio
import numpy as np
from math import cos, sin


Model_Params = {
    'T': 0.1,
    'm': 1,
    'k': 1,
    'b': 1,
    'l': 1,
    'Dx': 1,
    'Dy': 1,
    'Dz': 1,
    'Ixx': 1,
    'Iyy': 1,
    'Izz': 1
}


# use get_state_space to return A and B matrices for the formula
# x_k+1 = A*x_k + B*u_k
class Quadcopter_Model():
    def __init__(self, params:dict, linearized=False):
        self.params = params
        self.linearized = linearized
        self.A_base = self.init_system_matrix()
        self.B_base = self.init_input_matrix()
    

    # returns system and input matrices for Ax + Bu, computed online
    def get_state_space(self, state):
        # get copies of "base" A and B
        A = self.A_base
        B = self.B_base

        # compute angular acceleration portion of A
        J = self.get_jacobian(state)
        C = self.get_coriolis(state)
        A[15:18,9:12] = -np.inv(J) @ C

        # rotate acceleration portion of B
        B[12:15,0:4] = self.get_rotation_body_inertial(state) @ B[12:15,0:4]

        # compute torque with jacobian term
        J = self.get_jacobian(state)
        B[15:18,0:4] = inv(J) @ B[15:18,0:4]

        return A, B


    # converts body frame velocities to inertial frame
    def get_jacobian(self, state):
        phi, theta, psi, phi_d, theta_d, psi_d = state
        Ixx, Iyy, Izz = self.params['Ixx'], self.params['Iyy'], self.params['Izz']

        if self.linearized:
            return np.eye(3) * np.array([[Ixx],[Iyy],[Izz]], dtype=np.float32)
        else:
            J11 = Ixx
            J12 = 0
            J13 = -Ixx*sin(theta)
            J21 = 0
            J22 = Iyy*(cos(phi)**2) + Izz*(sin(phi)*2)
            J23 = (Iyy-Izz)*cos(phi)*sin(phi)*cos(theta)
            J31 = -Ixx*sin(theta)
            J32 = (Iyy-Izz)*cos(phi)*sin(phi)*cos(theta)
            J33 = Ixx*(sin(theta)**2) + Iyy*(sin(phi)**2)*(cos(theta)**2) + Izz*(cos(phi)**2)*cos(theta)**2
        
            return np.array([
                [J11, J12, J13],
                [J21, J22, J23],
                [J31, J32, J33]
            ], dtype=np.float32)


    # coriolis term that describes the effect of rotating frame within another frame
    def get_coriolis(self, state):
        if self.linearized:
            return np.zeros((3,3))
        else:
            phi, theta, psi, phi_d, theta_d, psi_d = state
            Ixx, Iyy, Izz = self.params['Ixx'], self.params['Iyy'], self.params['Izz']
        
            C11 = 0

            C12 = (Iyy-Izz)*(theta_d*cos(phi)*sin(phi) + psi_d*(sin(phi)**2)*cos(theta)) + \
                (Izz-Iyy)*psi_d*(cos(phi)**2)*cos(theta) - \
                Ixx*psi_d*cos(theta)

            C13 = (Izz-Iyy)*psi_d*cos(phi)*sin(phi)*(cos(theta)**2)

            C21 = (Izz-Iyy)*(theta_d*cos(phi)*sin(phi) + psi_d*sin(phi)*cos(theta)) + \
                (Iyy-Izz)*psi_d*(cos(phi)**2)*cos(theta) + \
                Ixx*psi_d*cos(theta)

            C22 = (Izz-Iyy)*phi_d*cos(phi)*sin(phi)

            C23 = -Ixx*psi_d*sin(theta)*cos(theta) + \
                Iyy*psi_d*(sin(phi)**2)*sin(theta)*cos(theta) + \
                Izz*psi_d*(cos(phi)**2)*sin(theta)*cos(theta)

            C31 = (Iyy-Izz)*psi_d*(cos(theta)**2)*sin(phi)*cos(phi) - \
                Ixx*theta_d*cos(theta)

            C32 = (Izz-Iyy)*(theta_d*cos(phi)*sin(phi)*sin(theta) + phi_d*(sin(phi)**2)*cos(theta)) + \
                (Iyy-Izz)*phi_d*(cos(phi)**2)*cos(theta) + \
                Ixx*psi_d*sin(theta)*cos(theta) - \
                Iyy*psi_d*(sin(phi)**2)*sin(theta)*cos(theta) - \
                Izz*psi_d*(cos(phi)**2)*sin(theta)*cos(theta)

            C33 = (Iyy-Izz)*phi_d*cos(phi)*sin(phi)*(cos(theta)**2) - \
                Iyy*theta_d*(sin(phi)**2)*cos(theta)*sin(theta) - \
                Izz*theta_d*(cos(phi)**2)*cos(theta)*sin(theta) + \
                Ixx*theta_d*cos(theta)*sin(theta)

            return np.array([
                [C11, C12, C13],
                [C21, C22, C23],
                [C31, C32, C33]
            ], dtype=np.float32)
    

    # rotation matrix converting x,y,z axes from body to inertial frame
    def get_rotation_body_inertial(self, state):
        phi, theta, psi, phi_d, theta_d, psi_d = state
        Rx = np.array(
            [1, 0, 0],
            [0, cos(phi), -sin(phi)],
            [0, sin(phi), cos(phi)]
        )
        Ry = np.array(
            [cos(theta), 0, sin(theta)],
            [0, 1, 0],
            [-sin(theta), 0, cos(theta) ]
        )
        Rz = np.array(
            [cos(psi), -sin(psi), 0],
            [sin(psi), cos(psi), 0],
            [0, 0, 1]
        )
        return Rz @ Ry @ Rx 


    # building the constant portions of system matrix
    def init_system_matrix(self):
        T, m = self.params['T'], self.params['m']
        Dx, Dy, Dz = self.params['Dx'], self.params['Dy'], self.params['Dz']
        A = np.zeros((18,18), dtype=np.float32)

        # adding discrete-time kinematics
        A[:12,:12] += np.eye(12)
        A[:12,6:18] += np.eye(12)*T
        A[:6,12:18] += np.eye(6)*0.5*T**2

        # adding drag portion
        A[12:15,6:9] = -1 * np.eye(3) * np.array([Dx,Dy,Dz]) / m
        return A
    

    # building constant portions of input matrix
    def init_input_matrix(self):
        k, m, l, b = self.params['k'], self.params['m'], self.params['l'], self.params['b']
        B = np.zeros((18, 4), dtype=np.float32)

        # adding acceleration portion without body-to-inertial-frame rotation
        B[12:15,0:4] = (k/m) * np.array(
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        , dtype=np.float32)

        # adding torque portion without jacobian term
        B[15:18,0:4] = np.array([
            [0, -l*k, 0, l*k],
            [-l*k, 0, l*k, 0],
            [-b, b, -b, b]
        ], dtype=np.float32)
        return B

