import control
import numpy as np

"""
Initialization of LQR gain matrix.
Given A and B matrices, as well as LQR parameters Q and R, returns LQR gain matrix Kc. 
"""
def Control_init(A,B,Q,R):

    #LQR gain matrix calculation
    Kc,_,_ = control.lqr(A,B,Q,R)

    return Kc

"""
Initialization of Kalman filter gain matrix.
Given A and C matrices, as well as covariance parameters for disturbance and noise, returns Kalman filter gain matrix Kf.
"""
def Kalman_init(A,C,Vn_cov=0.001,Vd_cov=0.001):

    #Covariance matrices
    assert type(Vd_cov) in [float, np.ndarray], "Noise covariance must be float or numpy array"
    if type(Vd_cov) == float:
        Vd = Vd_cov*np.identity(len(A))  # disturbance covariance
    else:
        Vd = np.array(Vd_cov)
    assert type(Vn_cov) in [float, np.ndarray], "Noise covariance must be float or numpy array"
    if type(Vn_cov) == float:
        Vn = Vn_cov*np.identity(len(A))    # noise covariance
    else:
        Vn = np.array(Vn_cov)

    #Kalman filter gain matrix calculation
    Kf_t,_,_=control.lqr(np.transpose(A),np.transpose(C),Vd,Vn)
    Kf=np.transpose(Kf_t)
    return Kf

"""
Initialization of state matrix X.
Given the starting state x0 and FE parameter Nt, returns X, a zero-matrix which keeps track of the system state.
"""
def X_init(x0,Nt):
    #Initialization of 'real system'
    X=np.zeros([len(x0),Nt+1])
    X[:,0]=x0

"""
Initialization of SCN Controller.
Given the SCN and FE parameters, returns SCN states with connections.
"""
def ControllerSCN_init(K,Nt,A,B,C,Kf,Kc,N,lam,bounding_box_factor,zero_init=True,x0=None,seed=0):

    np.random.seed(seed)

    D=np.random.randn(K,N) # N x K - Weights associated to each neuron
    D=D/np.linalg.norm(D,axis=0) #normalize
    D = D / bounding_box_factor # avoid too big discontinuities
    T = np.diag(D.T@D)/2

    # Initialize Voltage, spikes, rate
    V = np.zeros([N,Nt+1])
    s = np.zeros([N,Nt+1])
    r = np.zeros([N,Nt+1])

    # Set initial conditions
    if not zero_init:
        r[:,0] = np.array(np.linalg.pinv(D)@x0) # pseudo-inverse - "cheaty" way of getting the right firing rate
        V[:,0] = D.T@(x0-D@r[:,0])

    #We require an index for the weights, as the connections are only relevant for the first K/2 weights (the rest are for encoding the target state)
    i=int(K/2)

    # Network connections:
    # - fast
    O_f = D[:-i].T @ D[:-i]
    # - slow
    O_s = D[:-i].T @ (lam*np.identity(i) + A) @ D[:-i]
    # - rec. control
    O_c = -D[:-i].T @ B @ Kc @ D[:-i]
    # - ff. control
    F_c = D[:-i].T @ B @ Kc
    # - rec. kalman
    O_k = D[:-i].T @ Kf @ C @ D[:-i]
    # - ff kalman
    F_k = -D[:-i].T @ Kf

    return D,T,V,s,r,O_f,O_s,O_c,F_c,O_k,F_k
