import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys
import math


def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]


def plot(path=None,observedPath=None , estimate = None,path1=None,observedPath1=None , estimate1 = None, switch=1, v=None):
    plt.figure(figsize=(16,12))

    ax = plt.subplot(111)
    if path != None:
        x = np.zeros(len(path))
        y = np.zeros(len(path))
        for i in range(len(path)):
            x[i] = path[i][0][0]
            y[i] = path[i][1][0]
        plt.plot(x,y,'r-', label = "Actual Path")

    if observedPath != None:
        x = np.zeros(len(observedPath))
        y = np.zeros(len(observedPath))
        for i in range(len(observedPath)):
            x[i] = observedPath[i][0][0]
            y[i] = observedPath[i][1][0]
        plt.plot(x,y,'y-',label = "Observed Path")
    
    if estimate != None:
        x = np.zeros(len(estimate))
        y = np.zeros(len(estimate))
        for i in range(len(estimate)):
            x[i] = estimate[i][0][0][0]
            y[i] = estimate[i][0][1][0]
        plt.plot(x,y,'b-', label = "Estimate Path")
        plt.scatter(x,y,marker='.')

        if switch == 1:
            for i in range(len(estimate)):
                # if i %2 != 0:
                #     continue
                nstd = 2
                mean = estimate[i][0]
                cov = estimate[i][1][0:2,0:2]
                # print(cov)
                vals, vecs = eigsorted(cov)
                # print(vals,vecs)
                theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                w, h = 2 * nstd * np.sqrt(vals)
                # print(w,h,theta)
                ell = Ellipse(xy=(mean[0][0], mean[1][0]),
                            width=w, height=h,
                            angle=theta, color='blue')
                ell.set_facecolor('none')
                ax.add_artist(ell)

    if path1 != None:
        x = np.zeros(len(path1))
        y = np.zeros(len(path1))
        for i in range(len(path1)):
            x[i] = path1[i][0][0]
            y[i] = path1[i][1][0]
        plt.plot(x,y,'k-', label = "Actual Path1")

    if observedPath1 != None:
        x = np.zeros(len(observedPath1))
        y = np.zeros(len(observedPath1))
        for i in range(len(observedPath1)):
            x[i] = observedPath1[i][0][0]
            y[i] = observedPath1[i][1][0]
        plt.plot(x,y,'g-',label = "Observed Path1")
    
    if estimate1 != None:
        x = np.zeros(len(estimate1))
        y = np.zeros(len(estimate1))
        for i in range(len(estimate1)):
            x[i] = estimate1[i][0][0][0]
            y[i] = estimate1[i][0][1][0]
        plt.plot(x,y,'m', label = "Estimate Path1")
        plt.scatter(x,y,marker='.')

        if switch == 1:
            for i in range(len(estimate1)):
                # if i %2 != 0:
                #     continue
                nstd = 2
                mean = estimate1[i][0]
                cov = estimate1[i][1][0:2,0:2]
                # print(cov)
                vals, vecs = eigsorted(cov)
                # print(vals,vecs)
                theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                w, h = 2 * nstd * np.sqrt(vals)
                # print(w,h,theta)
                ell = Ellipse(xy=(mean[0][0], mean[1][0]),
                            width=w, height=h,
                            angle=theta, color='m')
                ell.set_facecolor('none')
                ax.add_artist(ell)
    
    if v!=None:
        for i in range(len(v)):
            plt.axvline(v[i])

    plt.legend()
    plt.show()




def plotVelocity(vel=None , estimate = None):
    plt.figure(figsize=(16,12))

    ax = plt.subplot(111)
    if vel != None:
        x = np.zeros(len(vel))
        y = np.zeros(len(vel))
        for i in range(len(vel)):
            x[i] = vel[i][2][0]
            y[i] = vel[i][3][0]
        plt.plot(x,y,'r-', label = "Actual vel")

    if estimate != None:
        x = np.zeros(len(estimate))
        y = np.zeros(len(estimate))
        for i in range(len(estimate)):
            x[i] = estimate[i][0][2][0]
            y[i] = estimate[i][0][3][0]
        plt.plot(x,y,'b-', label = "Estimate vel")
        plt.scatter(x,y)

    plt.legend()
    plt.show()



def initMotionModel(variance):
    A = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
    B = np.array([[0,0],[0,0],[1,0],[0,1]])
    R = np.array([[variance[0],0,0,0],[0,variance[1],0,0],[0,0,variance[2],0],[0,0,0,variance[3]]])
    return A,B,R

def initSensorModel(variance):
    C = np.array([[1,0,0,0],[0,1,0,0]])
    Q = variance*np.array([[1,0],[0,1]])
    return C,Q


def motionModel(state,u, A,B,R):
    avg = np.array([0,0,0,0])
    cov = R
    epsilon = np.random.multivariate_normal(avg,cov)
    temp = np.zeros((4,1))
    for i in range(4):
        temp[i][0] = epsilon[i]
    epsilon = temp
    nextState = np.matmul(A,state) + np.matmul(B,u) + epsilon
    return nextState

def sensorModel(state, C, Q):
    avg = np.array([0,0])
    cov = Q
    delta = np.random.multivariate_normal(avg,cov)
    temp = np.zeros((2,1))
    for i in range(2):
        temp[i][0] = delta[i]
    delta = temp
    
    observedState = np.matmul(C,state)+ delta
    return observedState



def kalmanFilter(mu,sigma,u,z,A,B,R,C,Q,switch=1):
    mu1 = np.matmul(A,mu)+np.matmul(B,u)
    sigma1 = np.matmul(np.matmul(A,sigma),np.transpose(A))+R

    k = np.matmul(np.matmul(sigma1,np.transpose(C)), np.linalg.inv(np.matmul(np.matmul(C,sigma1),np.transpose(C))+Q))

    if switch == 1:
        mu2 = mu1 + np.matmul(k,(z-np.matmul(C,mu1)))
        temp = np.matmul(k,C)
        l1,l2 = np.shape(temp)
        sigma2 = np.matmul((np.identity(l1)-temp), sigma1)
        return mu2,sigma2
    else:
        return mu1,sigma1


def dataAssociation(mu,sigma,u, A,B,R,C,Q,z1,
                                    mu1,sigma1,u1, A1,B1,R1,C1,Q1,z2):
    mu2 = np.matmul(A,mu)+np.matmul(B,u)
    mu2 = mu2[0:2,0:1]
    sigma2 = np.matmul(np.matmul(A,sigma),np.transpose(A))+R
    sigma2 = sigma2[0:2,0:2]

    d11 = np.matmul(np.transpose(z1-mu2),np.matmul(np.linalg.inv(sigma2),(z1-mu2)))
    d11 = math.sqrt(d11)
    d12 = np.matmul(np.transpose(z2-mu2),np.matmul(np.linalg.inv(sigma2),(z2-mu2)))
    d12 = math.sqrt(d12)

    mu3 = np.matmul(A1,mu1)+np.matmul(B1,u1)
    mu3 = mu3[0:2,0:1]
    sigma3 = np.matmul(np.matmul(A1,sigma1),np.transpose(A1))+R1
    sigma3 = sigma3[0:2,0:2]

    d21 = np.matmul(np.transpose(z1-mu3),np.matmul(np.linalg.inv(sigma3),(z1-mu3)))
    d21 = math.sqrt(d21)
    d22 = np.matmul(np.transpose(z2-mu3),np.matmul(np.linalg.inv(sigma3),(z2-mu3)))
    d22 = math.sqrt(d22)
    
    # if d11<d12:
    #     if d22<d21:
    #         return z1,z2
    #     else:
    #         return z1,z1
    # else:
    #     if d22<d21:
    #         return z2,z2
    #     else:
    #         return z2,z1
    if d11+d22 > d12+d21:
        return z2,z1

    return z1,z2

    
    

def main(part):
    # if part ==1:
    sim_len =   200
    # part 1
    A,B,R = initMotionModel([1,1,0.0001,0.0001])
    C,Q = initSensorModel(50)

    if part <= 3:
        initState = np.array([[10],[10],[5],[1]])
        
        path = [initState]
        observedPath = [sensorModel(initState,C,Q)]

        state = initState
        u = []
        for i in range(sim_len):
            u.append(np.array([[0.1],[-0.015]]))


        for i in range(sim_len):
            nextState = motionModel(state, u[i],A,B,R)
            path.append(nextState)
            observedState = sensorModel(nextState,C,Q)
            observedPath.append(observedState)
            state = nextState
    
    # print(observedPath)
    # plot(path,observedPath)

    # ----------------------------------------------------------------------------

    # part 2
        mu = np.array([[-100],[-75],[0],[0]])
        sigma = 0.0001*np.identity(4)
        dist = [(mu,sigma)]
        for i in range(sim_len):
            m1,s1 = kalmanFilter(mu,sigma,u[i],observedPath[i], A,B,R,C,Q)
            dist.append((m1,s1))
            mu = m1
            sigma = s1
            # print(mu)

        # print(dist[i][0][0])

    # ----------------------------------------------------------------------------

    # part 3
        plot(path=path,observedPath=observedPath, estimate = dist)


    # ----------------------------------------------------------------------------

    # part 4
    if part == 4:
        # Q=Q*50
        del1 = np.arange(0,sim_len,1)/10
        delx = np.sin(del1)
        dely = np.cos(del1)
        u = []
        for i in range(sim_len):
            u.append(np.array([[delx[i]],[dely[i]]]))
        
        # print(u)

        initState = np.array([[10],[10],[1],[1]])
        
        path = [initState]
        observedPath = [sensorModel(initState,C,Q)]

        state = initState

        for i in range(sim_len):
            nextState = motionModel(state, u[i],A,B,R)
            path.append(nextState)
            observedState = sensorModel(nextState,C,Q)
            observedPath.append(observedState)
            state = nextState
        
        mu = np.random.rand(4,1)
        sigma = 0.0001*np.identity(4)
        dist = [(mu,sigma)]
        for i in range(sim_len):
            m1,s1 = kalmanFilter(mu,sigma,u[i],observedPath[i], A,B,R,C,Q)
            dist.append((m1,s1))
            mu = m1
            sigma = s1

        plot(path=path,observedPath=observedPath, estimate = dist, switch=1)

        def plotError(p1,e1):
            e=[]
            for i in range(len(p1)):
                e.append(p1[i][0][0]-e1[i][0][0][0] + p1[i][1][0]-e1[i][0][1][0])
            
            x = []
            for i in range(len(e)):
                x.append(i)
            plt.bar(x,e)
            plt.show()


        plotError(path, dist)    
    # ----------------------------------------------------------------------------
    # part5

    if part == 5:
        # A,B,R = initMotionModel()
        # C,Q = initSensorModel()
        # Q = Q*0.10
        initState = np.array([[10],[10],[1],[1]])
        
        path = [initState]
        observedPath = [sensorModel(initState,C,Q)]

        state = initState
        u = []
        for i in range(sim_len):
            u.append(np.array([[0.1],[-0.015]]))


        for i in range(sim_len):
            nextState = motionModel(state, u[i],A,B,R)
            path.append(nextState)
            observedState = sensorModel(nextState,C,Q)
            observedPath.append(observedState)
            state = nextState
        
        mu = np.array([[0],[0],[0],[0]])
        sigma = 0.0001*np.identity(4)
        dist = [(mu,sigma)]
        for i in range(sim_len):
            m1,s1 = kalmanFilter(mu,sigma,u[i],observedPath[i], A,B,R,C,Q)
            dist.append((m1,s1))
            mu = m1
            sigma = s1
            # print(mu)

        plot(path=path,observedPath=observedPath, estimate = dist, switch=0)

    # ----------------------------------------------------------------------------
    # part6
    if part == 6:
        del1 = np.arange(0,sim_len,1)/10
        delx = np.sin(del1)
        dely = np.cos(del1)
        u = []
        for i in range(sim_len):
            u.append(np.array([[delx[i]],[dely[i]]]))
        
        # print(u)

        initState = np.array([[10],[10],[1],[1]])
        
        path = [initState]
        observedPath = [sensorModel(initState,C,Q)]

        state = initState

        for i in range(sim_len):
            nextState = motionModel(state, u[i],A,B,R)
            path.append(nextState)
            observedState = sensorModel(nextState,C,Q)
            observedPath.append(observedState)
            state = nextState
        
        mu = np.array([[-50],[100],[0],[0]])
        sigma = np.identity(4)
        sigma[0][0] = 10000
        sigma[1][1] = 10000
        dist = [(mu,sigma)]
        for i in range(sim_len):
            m1,s1 = kalmanFilter(mu,sigma,u[i],observedPath[i], A,B,R,C,Q)
            dist.append((m1,s1))
            mu = m1
            sigma = s1

        plot(path=path,observedPath=observedPath, estimate = dist)

    # ----------------------------------------------------------------------------
    # part 7
    if part == 7:
        sim_len = 100
        del1 = np.arange(0,sim_len,1)/10
        delx = np.sin(del1)
        dely = np.cos(del1)
        u = []
        for i in range(sim_len):
            u.append(np.array([[delx[i]],[dely[i]]]))
        
        # print(u)

        initState = np.array([[10],[10],[1],[1]])
        
        path = [initState]
        observedPath = [sensorModel(initState,C,Q)]

        state = initState

        for i in range(sim_len):
            nextState = motionModel(state, u[i],A,B,R)
            path.append(nextState)
            observedState = sensorModel(nextState,C,Q)
            observedPath.append(observedState)
            state = nextState
        
        mu = np.array([[0],[0],[0],[0]])
        sigma = 0.0001*np.identity(4)
        dist = [(mu,sigma)]
        v=[]
        for i in range(sim_len):
            if (i>=10 and i<20) or (i>=30 and i<40):
                m1,s1 = kalmanFilter(mu,sigma,u[i],observedPath[i], A,B,R,C,Q,switch=0)
                dist.append((m1,s1))
                mu = m1
                sigma = s1
                if i==10:
                    v.append(m1[0][0])
                if i==19:
                    v.append(m1[0][0])
                if i==30:
                    v.append(m1[0][0])
                if i==39:
                    v.append(m1[0][0])
            else:
                m1,s1 = kalmanFilter(mu,sigma,u[i],observedPath[i], A,B,R,C,Q)
                dist.append((m1,s1))
                mu = m1
                sigma = s1
        
        plot(path=path, estimate = dist, v=v)
    
    if part == 8:
        sim_len =400
        u = []
        for i in range(sim_len):
            u.append(np.array([[0.1],[-0.01]]))
                
        # del1 = np.arange(0,sim_len,1)/10
        # delx = np.sin(del1)
        # dely = np.cos(del1)
        # u = []
        # for i in range(sim_len):
        #     u.append(np.array([[delx[i]],[dely[i]]]))
        # # print(u)

        initState = np.array([[10],[10],[1],[1]])
        
        path = [initState]
        observedPath = [sensorModel(initState,C,Q)]

        state = initState

        for i in range(sim_len):
            nextState = motionModel(state, u[i],A,B,R)
            path.append(nextState)
            observedState = sensorModel(nextState,C,Q)
            observedPath.append(observedState)
            state = nextState
        
        mu = np.array([[0],[0],[0],[0]])
        sigma = 0.0001*np.identity(4)
        dist = [(mu,sigma)]
        for i in range(sim_len):
            m1,s1 = kalmanFilter(mu,sigma,u[i],observedPath[i], A,B,R,C,Q)
            dist.append((m1,s1))
            mu = m1
            sigma = s1

        plotVelocity(vel=path, estimate = dist)
    
    if part == 9:
        # sim_len=180
        A,B,R = initMotionModel([1,1,0.0001,0.0001])
        C,Q = initSensorModel(100)
        initState = np.array([[-100],[10000],[0.1],[-100]])
        
        path = [initState]
        observedPath = [sensorModel(initState,C,Q)]

        state = initState
        u = []
        for i in range(sim_len):
            u.append(np.array([[0.1],[0]]))


        for i in range(sim_len):
            nextState = motionModel(state, u[i],A,B,R)
            path.append(nextState)
            observedState = sensorModel(nextState,C,Q)
            observedPath.append(observedState)
            state = nextState
        


        A1,B1,R1 = initMotionModel([1.5,1.1,0.0001,0.0001])
        C1,Q1 = initSensorModel(100)

        initState1 = np.array([[-5000],[-10000],[100],[120]])
        
        path1 = [initState1]
        observedPath1 = [sensorModel(initState1,C1,Q1)]

        state1 = initState1
        u1 = []
        for i in range(sim_len):
            u1.append(np.array([[0.1],[0]]))


        for i in range(sim_len):
            nextState = motionModel(state1, u1[i],A1,B1,R1)
            path1.append(nextState)
            observedState = sensorModel(nextState,C1,Q1)
            observedPath1.append(observedState)
            state1 = nextState

        mu = np.array([[-100],[10000],[0],[0]])
        sigma = 0.0001*np.identity(4)
        dist = [(mu,sigma)]
        mu1 = np.array([[-5000],[-10000],[0],[0]])
        sigma1 = 0.02*np.identity(4)
        dist1 = [(mu1,sigma1)]
        
        for i in range(sim_len):
            z1,z2 = observedPath[i],observedPath1[i]

            z1,z2 = dataAssociation(mu,sigma,u[i], A,B,R,C,Q,z1,
                                    mu1,sigma1,u1[i], A1,B1,R1,C1,Q1,z2)
            
            m1,s1 = kalmanFilter(mu,sigma,u[i],z1, A,B,R,C,Q)
            dist.append((m1,s1))
            mu = m1
            sigma = s1

            m1,s1 = kalmanFilter(mu1,sigma1,u1[i],z2, A1,B1,R1,C1,Q1)
            dist1.append((m1,s1))
            mu1 = m1
            sigma1 = s1

        plot(path=path,observedPath=observedPath, estimate = dist,path1=path1,observedPath1=observedPath1, estimate1 = dist1,switch=0)
    
    plt.show()


for i in range(9):
    main(i+1)