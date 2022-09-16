import matplotlib.pyplot as plt
import numpy as np

T_all = 1
Ts = T_all/4


def positive_calculation(self, t):

    z_start=-0.05
    z_end=0.05
    y_start=-0.35
    h=0.05

    t_time = t%T_all

    if(t_time<=Ts):
        sigma=2*np.pi*t_time/Ts;  
        z=(z_end-z_start)*((sigma-np.sin(sigma))/(2*np.pi))+z_start
        y=h*(1-np.cos(sigma))/2+y_start
    else:
        z=(-(z_end-z_start)/(T_all-Ts))*(t_time-Ts)+z_end
        y=y_start
    return (z, y)

def negative_calculation(self, t):

    z_start=0.05
    z_end=-0.05
    y_start=-0.35
    h=0.05

    t_time = t%T_all

    if(t_time<=Ts):
        sigma=2*np.pi*t_time/Ts;  
        z=(z_end-z_start)*((sigma-np.sin(sigma))/(2*np.pi))+z_start
        y=h*(1-np.cos(sigma))/2+y_start
    else:
        z=(-(z_end-z_start)/(T_all-Ts))*(t_time-Ts)+z_end
        y=y_start
    return (z, y)