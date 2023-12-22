import pandas as pd
import numpy as np
import ast
import json
from Trajectory_planning import plan_cubic_trajectory, plan_quintic_trajectory

f = open('Config.json')             #open the json file 
Json = json.load(f)                 #initialize Json 

df= pd.read_excel('Book1.xlsx', usecols=range(6,7))       #access the Excel file, expectially the column 

numpy_array= df.to_numpy()          #Convert the dataframe to numpy array
numpy_array = [ast.literal_eval(element[0]) for element in numpy_array] #[1,2,3]    #Convert the object to array using ast

mvag = np.zeros((Json['Sample'], Json['N']))            #Create a Matrix consist specific number of column and row. Column = Number of classification(x, y,z,....). Row = Sample size(0,0,0,0,....)
M = np.zeros((Json['N'], 4, 4))         #Create a 3D array for the matrix. Json['N'] indicates the number of classification. If 3 = [x,y,z], if 2 = [x,y]

temp_b_cubic = np.zeros((Json['N'],4,1))            #[v0, a0, vt, at] cubic trajectory
temp_b_cubic = temp_b_cubic.reshape((Json['N'],1,-1))           #rearrange the array horizontally

temp_b_quantic= np.zeros((Json['N'],6,1))           #[v0, a0, j0, vt, at, jt]  
temp_b_quantic = temp_b_quantic.reshape((Json['N'],1,-1))           #rearrange the array horizontally       

velocity_prev = np.zeros(Json['N'])             #Create a empty array for the velocity        
acceleration_prev = np.zeros(Json['N'])         #Create a empty array for the acceleration
mean_prev = np.zeros(Json['N'])

t0 = 0     #initial time
t1 = 0.02  #timestep in s
dt = 0.001 #increment time

########################################################################################
#  Function Name : push()
#  Function use  : to push new incoming data in to the matrix 
#  Input         : (variable) mvag: NDArray[float64] , (variable) numpy_array: list[Any]
#  Output        : (variable) mvag: NDArray[float64]
########################################################################################
def push(x, y): 
    x[:-1] = x[1:]
    x[-1:] = y
    return x

for i in range (len(numpy_array)):

    push(mvag, numpy_array[i][0:Json['N']])
    
    mean = np.mean(mvag, axis= 0) 
 
    velocity_new = 0.5*Json['max_vel']*mean**2
    accleration_new = Json['max_acc']*mean
    

    for j in range (Json['N']):
          
        temp_b_cubic[j] = ([velocity_prev[j], acceleration_prev[j], velocity_new[j], accleration_new[j]]) #arrange the 
        # temp_b_quantic[j] = ([velocity_prev[j], acceleration_prev[j], mean_prev[j], velocity_new[j], accleration_new[j], mean[j]])

    velocity_prev = velocity_new # velocity_prev = velocity # set initial velocity for next iteration to the target vel of this iteration
    accleration_prev = accleration_new
    # mean_prev=mean 

    time = np.array([t0, t1, dt])

    trajectory1 = plan_cubic_trajectory(temp_b_cubic, time)
    # trajectory = plan_quintic_trajectory(temp_b_quantic, time)

    t0 = t1+dt
    t1= t1+0.02

    #print(trajectory)
        



 


