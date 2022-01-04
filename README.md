# State-Space-Estimation
## Implementation of HMM filter 

Consider the problem of localizing an aerial vehicle that is flying at a constant height while
surveying an area. Please see figure below. Assume a uniform discretization for the region with grid
cells of size 1m × 1m. At each time step, the robot can execute one of four actions: up, down, left or
right to an adjacent grid cell, selected with likelihoods 0.4, 0.1, 0.2 and 0.3 respectively. The area is
instrumented with four sensors positioned at the indicated grid cells (see figure). Each sensor reports
the discrete presence or absence of a target without reporting any other information such as the target’s
position in the grid. Sensor observations are stochastic. The likelihood of reporting the presence of a
target is shown in the figure (right). Sensor observations are generated independently at each time step
and each sensor reports independent of all other sensors. Our goal is to estimate the position of the
agent in the grid given the assumptions above.


## Kalman Filter
Consider an airplane flying in a (x, y)-plane. A noisy sensor (e.g., a radar) provides measurements zt = (x0t,y0t). 
The plane is controlled by providing velocity increments ut = [δx˙ t, δy˙t], which
get added to the velocity components ˙xt and ˙yt at each time step. The uncertainty in the motions is
characterized by a presence of Gaussian noise t ∼ N (0, R). Similarly, the measurement uncertainty is
characterized by presence of Gaussian noise δt ∼ N (0, Q). Our goal is to estimate its positions [xt, yt]
and velocities [ ˙xt, y˙t] at time instant t from noisy observations [x0t, y0t]
