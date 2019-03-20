# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv
from scipy.linalg import logm, norm, sqrtm
from ControlParams import *
from pyquaternion import Quaternion
import quadprog
import copy


def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2


    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

def robotParams():
    I3 = np.eye(3)
    ex = I3[:,0]
    ey = I3[:,1]
    ez = I3[:,2]
    
    h1 = ez
    h2 = ey
    h3 = ey
    h4 = ex
    h5 = ey
    h6 = ex
    P = np.array([[0,0,0], [0.32, 0, 0.78], [0, 0, 1.075], [0, 0, 0.2], [1.142, 0, 0], [0.2, 0, 0], [0,0,0]]).T
    q = np.zeros((6, 1))
    H = np.array([h1, h2, h3, h4, h5, h6]).T
    ttype = np.zeros((1, 6))

    n = 6
    
    dq_bounds = np.array([[100,110], [90,90], [90,90], [170,190], [120,140], [190,235]]).T
    dq_bounds = dq_bounds*np.pi/180
    
    return ex,ey,ez,n,P,q,H,ttype,dq_bounds


    
# find closest rotation matrix 
# A=A*inv(sqrt(A'*A))   
def Closest_Rotation(R):
    R_n = np.dot(R, inv(sqrtm(np.dot(R.T, R))))
    
    return R_n

# ROT Rotate along an axis h by q in radius
def rot(h, q):
    h=h/norm(h)
    R = np.eye(3) + np.sin(q)*hat(h) + (1 - np.cos(q))*np.dot(hat(h), hat(h))
    
    return R

def hat(h):
    h_hat = np.array([[0, -h[2], h[1]], [h[2], 0, -h[0]], [-h[1], h[0], 0]])
    
    return h_hat
    
def fwdkin_alljoints(q, ttype, H, P, n):
    R=np.eye(3)
    p=np.zeros((3,1))
    RR = np.zeros((3,3,n+1))
    pp = np.zeros((3,n+1))
    
    for i in range(n):
        h_i = H[0:3,i]
       
        if ttype[0][i] == 0:
        #rev
            pi = P[0:3,i].reshape(3, 1)
            p = p+np.dot(R,pi)
            Ri = rot(h_i,q[i])
            R = np.dot(R,Ri)
            R = Closest_Rotation(R)
        elif ttype[i] == 1: 
        #pris
            pi = (P[:,i]+q[i]*h_i).reshape(3, 1)
            p = p+np.dot(R,pi)
        else: 
	    # default pris
	        pi = (P[:,i]+q[i]*h_i).reshape(3, 1)
	        p = p+np.dot(R,pi)
        
        pp[:,[i]] = p
        RR[:,:,i] = R
    
    # end effector T
    p=p+np.dot(R, P[0:3,n].reshape(3, 1))
    pp[:,[n]] = p
    RR[:,:,n] = R
    
    return pp, RR

def getJacobian(q,ttype,H,P,n):
    num_joints = len(q)

    P_0_i = np.zeros((3,num_joints+1))
    R_0_i = np.zeros((3,3,num_joints+1))


    P_0_i,R_0_i=fwdkin_alljoints(q,ttype,H,P,n)
    
    P_0_T = P_0_i[:,num_joints]

    J = np.zeros((6,num_joints))
    
    for i in range(num_joints):
        if ttype[0][i] == 0:
            J[:,i] = np.hstack((np.dot(R_0_i[:,:,i],H[:,i]), np.dot(hat(np.dot(R_0_i[:,:,i], H[:,i])), (P_0_T - P_0_i[:,i]))))
    """ """
    
    return J


def getqp_H(J, vr, vp, er, ep):
    n = 6

    H1 = np.dot(np.hstack((J,np.zeros((n,2)))).T,np.hstack((J,np.zeros((n,2)))))
    
    tmp = np.vstack((np.hstack((np.hstack((np.zeros((3,n)),vr)),np.zeros((3,1)))),np.hstack((np.hstack((np.zeros((3,n)),np.zeros((3,1)))),vp)))) 
    H2 = np.dot(tmp.T,tmp)

    H3 = -2*np.dot(np.hstack((J,np.zeros((n,2)))).T, tmp)
    H3 = (H3+H3.T)/2;
    
    tmp2 = np.vstack((np.array([0,0,0,0,0,0,np.sqrt(er),0]),np.array([0,0,0,0,0,0,0,np.sqrt(ep)])))
    H4 = np.dot(tmp2.T, tmp2)

    H = 2*(H1+H2+H3+H4)

    return H

def getqp_f(er, ep):
    f = -2*np.array([0,0,0,0,0,0,er,ep]).reshape(8, 1)
    
    return f

def inequality_bound(h,c,eta,epsilon,e):
    sigma = np.zeros((h.shape))
    h2 = h - eta
    sigma[np.array(h2 >= epsilon)] = -np.tan(c*np.pi/2)
    sigma[np.array(h2 >= 0) & np.array(h2 < epsilon)] = -np.tan(c*np.pi/2/epsilon*h2[np.array(h2 >= 0) & np.array(h2 < epsilon)])
    sigma[np.array(h >= 0) & np.array(h2 < 0)] = -e*h2[np.array(h >= 0) & np.array(h2 < 0)]/eta
    sigma[np.array(h < 0)] = e
    
    return sigma


def getqp_H_cam(J, UV):
    n = J.shape[0]

    H1 = np.dot(np.hstack((J,np.zeros((n,1)))).T,np.hstack((J,np.zeros((n,1)))))
    
    tmp = np.vstack((np.zeros((6,n)),UV.T)).T
    H2 = np.dot(tmp.T,tmp)

    H3 = -2*np.dot(np.hstack((J,np.zeros((n,1)))).T, tmp)
    H3 = (H3+H3.T)/2;


    H = 2*(H1+H2+H3)

    
    return H

def getqp_f_cam():
    f = -2.0*np.array([0,0,0,0,0,0,0]).reshape(7,)
    
    return f
    

def QP_abbirb6640(q,v):
    #Init the joystick

    
    # Initialize Robot Parameters    
    ex,ey,ez,n,P,q_ver,H,ttype,dq_bounds = robotParams()
    # joint limits
    lower_limit = np.transpose(np.array([-170*np.pi/180, -65*np.pi/180, -np.pi, -300*np.pi/180, -120*np.pi/180, -2*np.pi]))
    upper_limit = np.transpose(np.array([170*np.pi/180, 85*np.pi/180, 70*np.pi/180, 300*np.pi/180, 120*np.pi/180, 2*np.pi]))
    
    # Initialize Control Parameters
    # initial joint angles
    

    pos_v = np.zeros((3, 1))
    ang_v = np.array([1,0,0,0])
    dq = np.zeros((int(n),1))

    # inequality constraints
    h = np.zeros((15, 1))
    sigma = np.zeros((13, 1))
    dhdq = np.vstack((np.hstack((np.eye(6), np.zeros((6, 1)), np.zeros((6, 1)))), np.hstack((-np.eye(6), np.zeros((6, 1)), np.zeros((6, 1)))), np.zeros((1, 8))))

    # velocities
    w_t = np.zeros((3, 1))
    v_t = np.zeros((3, 1))
    
    # keyboard controls
    # define position and angle step
    inc_pos_v = 0.01 # m/s
    inc_ang_v = 0.5*np.pi/180 # rad/s

    # optimization params
    er = 0.05
    ep = 0.05
    epsilon = 0 # legacy param for newton iters
    
    # parameters for inequality constraints
    c = 0.5
    eta = 0.1
    epsilon_in = 0.15
    E = 0.005

    pp,RR = fwdkin_alljoints(q,ttype,H,P,n)
    orien_tmp = Quaternion(matrix=RR[:, :, -1])
    orien_tmp = np.array([orien_tmp[0], orien_tmp[1], orien_tmp[2], orien_tmp[3]]).reshape(1, 4)
    
    # create a handle of these parameters for interactive modifications
    obj = ControlParams(ex,ey,ez,n,P,H,ttype,dq_bounds,q,dq,pp[:, -1],orien_tmp,pos_v,ang_v.reshape(1, 4),w_t,v_t,epsilon,inc_pos_v,inc_ang_v,0,er,ep,0)

    
    J_eef = getJacobian(obj.params['controls']['q'], obj.params['defi']['ttype'], obj.params['defi']['H'], obj.params['defi']['P'], obj.params['defi']['n'])
    
    
    # desired rotational velocity
    vr = v[0:3]
    
    # desired linear velocity
    vp = v[3:6]
                
    Q = getqp_H(J_eef, vr.reshape(3, 1),  vp.reshape(3, 1), obj.params['opt']['er'], obj.params['opt']['ep']) 
    
    # make sure Q is symmetric
    Q = 0.5*(Q + Q.T)
    
    f = getqp_f(obj.params['opt']['er'], obj.params['opt']['ep'])
    f = f.reshape((8, ))

    
    # bounds for qp
    if obj.params['opt']['upper_dq_bounds']:
        bound = obj.params['defi']['dq_bounds'][1, :]
    else:
        bound = obj.params['defi']['dq_bounds'][0, :]

    LB = np.vstack((-0.1*bound.reshape(6, 1),0,0))
    UB = np.vstack((0.1*bound.reshape(6, 1),1,1))
            
    # inequality constrains A and b
    h[0:6] = obj.params['controls']['q'] - lower_limit.reshape(6, 1)
    h[6:12] = upper_limit.reshape(6, 1) - obj.params['controls']['q']

    sigma[0:12] = inequality_bound(h[0:12], c, eta, epsilon_in, E)
    

    A = np.vstack((dhdq,np.eye(8), -np.eye(8)))
    b = np.vstack((sigma, LB, -UB))

    # solve the quadprog problem
    dq_sln = quadprog.solve_qp(Q, -f, A.T, b.reshape((29, )))[0]

        
    if len(dq_sln) < obj.params['defi']['n']:
        obj.params['controls']['dq'] = np.zeros((6,1))
        V_scaled = 0
        print 'No Solution'
        dq_sln = np.zeros((int(n),1))
    else:
        obj.params['controls']['dq'] = dq_sln[0: int(obj.params['defi']['n'])]
        obj.params['controls']['dq'] = obj.params['controls']['dq'].reshape((6, 1))
        #print dq_sln
#        V_scaled = dq_sln[-1]*V_desired
#        vr_scaled = dq_sln[-2]*vr.reshape(3,1)
        
        #print np.dot(np.linalg.pinv(J_eef),v)
     
    return dq_sln[0:6]
                  
                  
def QP_Cam(J,UV):

    Q = getqp_H_cam(J, UV)
    # make sure Q is symmetric
    Q = nearestPD(Q)
    f = getqp_f_cam()


    LB = np.vstack((-0.1*np.ones([6,1]),1.0))
    UB = np.vstack((0.1*np.ones([6,1]),2.0))
             

    A = np.vstack((np.eye(7), -np.eye(7)))
    b = np.vstack((LB, -UB))

    # solve the quadprog problem   
    dq_sln = quadprog.solve_qp(Q, -f, A.T, b.reshape((14, )))[0]

        
    return dq_sln[0:6]
                           

def QP_TwoCam(alpha,J1,UV1,J2,UV2):

    Q1 = getqp_H_cam(J1, UV1)
    Q2 = getqp_H_cam(J2, UV2)
    Q = alpha*nearestPD(Q1)+(1-alpha)*nearestPD(Q2)
    Q = nearestPD(Q)
    
    f = getqp_f_cam()


    LB = np.vstack((-0.1*np.ones([6,1]),1.0))
    UB = np.vstack((0.1*np.ones([6,1]),2.0))
             

    A = np.vstack((np.eye(7), -np.eye(7)))
    b = np.vstack((LB, -UB))

    # solve the quadprog problem    
    dq_sln = quadprog.solve_qp(Q, -f, A.T, b.reshape((14, )))[0]

        
    return dq_sln[0:6]


def trapgen(xo,xf,vo,vf,vmax,amax,dmax,t):

#Generate trapezoidal velocity motion profile.
#
#inputs:
#xo: initial position
#xf: final position
#vo: initial velocity
#vf: final velocity
#vmax: velocity limit
#amax: upper acceleration limit (magnitude)
#dmax: lower acceleration limit (magnitude)
#t: sample time
#
#outputs:
#x: position at sample time t
#v: velocity at sample time t
#a: acceleration at sample time t
#ta: first switching time
#tb: second switching time
#tf: final time

  # vo and vf must be less than vmax
    if (abs(vo)>=abs(vmax))|(abs(vf)>=abs(vmax)):
        print('vo or vf too large')
    
    vmax=np.sign(xf-xo)*vmax
    
    if xf>xo:
        am1=abs(amax)
        am2=-abs(dmax)
    else:
        am1=-abs(dmax)
        am2=abs(amax)
        
    ta=abs((vmax-vo)/am1)
    tf_tb=(vf-vo-am1*ta)/am2
    print (am1*ta+vo)
    tf=(xf-xo+.5*am1*ta**2-.5*am2*tf_tb**2)/(am1*ta+vo)
    tb=tf-tf_tb
    
    if ((tf<2*ta)|(tb<ta)):
        #tapoly=[1, 2*vo/am1 ((vf**2-vo**2)/2/am2+xo-xf)*2/(1-am1/am2)/am1]
        ta = -vo/am1 + np.sqrt((vo/am1)**2-	(2/(1-am1/am2)/am1)*((vf-vo)**2/2/am2+xo-xf))
        tf=(vf-vo-am1*ta)/am2+ta
        tb=ta
    
    if t<ta:
        a=am1
        v=am1*t+vo
        x=.5*am1*t**2+vo*t+xo
    elif t<tb:
        a=0
        v=am1*ta+vo;
        x=am1*(t*ta-.5*ta**2)+vo*t+xo;
    elif t<=tf:
        a=am2
        v=am2*(t-tb)+am1*ta+vo
        x=am1*(-.5*ta**2+t*ta)+am2*(.5*t**2+.5*tb**2-tb*t)+vo*t+xo
    else:
        a=0
        v=0
        x=xf

    return x,v,a,ta,tb,tf   



def sort_corners(corners,ids):
    #Sort corners and ids according to ascending order of ids
    corners_original = copy.deepcopy(corners)
    ids_original = np.copy(ids)
    sorting_indices = np.argsort(ids_original,0)
    ids_sorted = ids_original[sorting_indices]
    ids_sorted = ids_sorted.reshape([len(ids_original),1])
    combined_list = zip(np.ndarray.tolist(ids.flatten()),corners_original)
    combined_list.sort(key=lambda tup: tup[0])
    corners_sorted = [x for y,x in combined_list]
    ids = np.copy(ids_sorted)
    corners = copy.deepcopy(corners_sorted)
    
    #Remove ids and corresponsing corners not in range (Parasitic detections in random locations in the image)       
    mask = np.ones(ids.shape, dtype=bool)        
    ids = ids[mask]        
    corners = np.array(corners)        
    corners = corners[mask.flatten(),:]        
    corners = list(corners)
    
    return corners,ids
     
#if __name__ == '__main__':
#    q = np.array([0,0,0,0,np.pi/2,0]).reshape(6, 1)
#    v = np.array([0,0,0,0,0,0.1])
#    a=QP_abbirb6640(q,v)
#    print a
    
