from scipy.spatial.transform import Rotation
from math import atan2, asin, degrees
q0, q1, q2, q3 = [ 1,0.5,0,1] #wxyz
rot = Rotation.from_quat([[q1,q2,q3,q0]])
print(rot.as_euler('zxz', degrees=True))
phi = atan2((2*(q0*q1+q2*q3)), (1-2*(pow(q1, 2)+pow(q2, 2))))

theta = asin(2*(q0*q2-q1*q1))

psi = atan2((2 * (q0 * q3 + q1 * q2)),
            (1 - 2 * (pow(q2, 2) + pow(q3, 2))))

print(degrees(phi), degrees(theta), degrees(psi))