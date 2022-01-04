import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def sample_sensor_data(sensor_grid, position):
    sensor_out = []
    x,y = position
    for i in range(4):
        r = random.uniform(0,1)
        if r<= sensor_grid[i][x][y]:   # due to origin shifting
            sensor_out.append(1)   # shows present
        else:
            sensor_out.append(0)   # shows absent
    return sensor_out

def simulate(sensor_grid,p_actions,T, initial_pos, l, b):
    positions = [initial_pos]
    actions =[]
    sensors_data = []
    sensors_data.append(sample_sensor_data(sensor_grid,initial_pos))


    for i in range(T):
        # sample an action and apply it
        r = random.uniform(0,1)      
        action = None
        next_pos = None
        if r<=p_actions[0]:
            action = 'up'
            next_pos = (min(b-1,positions[-1][0]+1) ,positions[-1][1] )

        elif r<= (p_actions[0]+p_actions[1]):
            action = 'down'
            next_pos = (max(0,positions[-1][0]-1), positions[-1][1] )

        elif r<= (p_actions[0]+ p_actions[1]+ p_actions[2]):
            action = 'left'
            next_pos = (positions[-1][0],max(0,positions[-1][1]-1))

        elif r<= 1:
            action = 'right'
            next_pos = (positions[-1][0],min(l-1,positions[-1][1]+1))

        actions.append(action)
        positions.append(next_pos)        
        
        # sample sensor data according to next pos
        # remember sensors are independent so sample again and again
        sensors_data.append(sample_sensor_data(sensor_grid,next_pos))

    return actions, sensors_data, positions


def getProbability(pos, sensor_grid, z):
    x,y = pos
    p=1
    for i in range(4):
        temp = sensor_grid[i][x][y]
        if z[i] ==0:
            temp = 1-temp
        p=p*temp
    return p

def Bayes_filter(belief_grid, sensor_grid, d, flag):
    n=0
    if flag == 0:   # perceptual data item z
        z = d
        for i in range(30):
            for j in range(30):
                p = getProbability((i,j),sensor_grid, z)
                belief_grid[i][j] = p*belief_grid[i][j]
                n = n + belief_grid[i][j]

        return belief_grid/n
        
    else:           # action data item 
        u = d
        temp = np.copy(belief_grid)
        for i in range(30):
            for j in range(30):
                belief_grid[i][j] = 0
                if u == 'up':
                    if i>0:
                        belief_grid[i][j] = temp[i-1][j]
                elif u == 'down':
                    if i<29:
                        belief_grid[i][j] = temp[i+1][j]

                elif u == 'right':
                    if j>0:
                        belief_grid[i][j] = temp[i][j-1]
                elif u == 'left':
                    if j<29:
                        belief_grid[i][j] = temp[i][j+1]
        
        return belief_grid
            

def generateSensorGrid(sensor_pos):
    grid = np.zeros((30,30))
    x,y = sensor_pos
    c=0.9
    for i in range(5):
        for j in range(-1*i,i+1):
            grid[x-i][y+j] = c-(0.1*i)
            grid[x+i][y+j] = c-(0.1*i)
            grid[x+j][y-i] = c-(0.1*i)
            grid[x+j][y+i] = c-(0.1*i)
    return grid
    

def plot(belief_grid):
    fig = plt.figure(figsize=(6, 6))

    plt.imshow(belief_grid,cmap='gray')
    plt.show()

# def ani(arr):
#     fig, ax = plt.subplots(figsize=(8, 8))


#     def update(i):
#         ax.imshow(arr[i], cmap='gray')


#     anim = FuncAnimation(fig, update, frames=np.arange(0, 25), interval=200)
#     # anim.save('colour_rotation.gif', dpi=80, writer='imagemagick')
#     plt.show()
#     plt.close()

def main():
    T = 50
    l,b = 30,30
    sensor_pos = [(15,8),(22,15),(15,15),(15,22)]
    
    sensor_grid = [generateSensorGrid((15,8)),generateSensorGrid((22,15)),generateSensorGrid((15,15)),generateSensorGrid((15,22))]
    
    p_actions = [0.4,0.1,0.2,0.3] # up, down, left, right respectively
    
    initial_pos = (10,10)
    actions, sensors_data, positons = simulate(sensor_grid,p_actions,T, initial_pos,l,b)
    print(actions)
    print(sensors_data)
    print(positons)
    print()

    belief_grid = np.ones((30,30))
    belief_grid = belief_grid/900
    belief_grid = Bayes_filter(belief_grid, sensor_grid, sensors_data[0],0)
    plot(belief_grid)
    # arr =[belief_grid]

    for i in range(T):
        belief_grid = Bayes_filter(belief_grid, sensor_grid, actions[i],1)
        belief_grid = Bayes_filter(belief_grid, sensor_grid, sensors_data[i+1],0)
        plot(belief_grid)
        # arr.append(np.copy(belief_grid))
    
    # ani(arr)
    
        
    


    
main()