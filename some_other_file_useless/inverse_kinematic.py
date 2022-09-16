import numpy as np

def inverse_kinematic_zzz(x,y,z):
    h=0
    hu=0.26
    hl=0.25
    dxy=np.sqrt(y**2+x**2)
    lxy=np.sqrt(dxy**2-h**2)
    gamma_xy=-np.arctan(x/y)
    gamma_h_offset=-np.arctan(h/lxy)
    gamma=gamma_xy-gamma_h_offset

    lyzp=np.sqrt(lxy**2+z**2)
    n=(lyzp**2-hl**2-hu**2)/(2*hu)
    beta=-np.arccos(n/hl) #beta总是小于0

    alfa_yzp=-np.arctan(z/lxy)
    alfa_off=np.arccos((hu+n)/lyzp)
    alfa=alfa_yzp+alfa_off

    return [gamma, alfa, beta]
    #输出角度为弧度