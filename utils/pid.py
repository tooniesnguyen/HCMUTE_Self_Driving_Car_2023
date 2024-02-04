import numpy as np
import time


pre_t = time.time()
err_arr = np.zeros(5)

def PID(err, Kp, Ki, Kd, mode):
    global pre_t
    err_arr[1:] = err_arr[0:-1]
    err_arr[0] = err
    delta_t = time.time() - pre_t
    pre_t = time.time()

    if Ki >= 2000:
        Ki = 2000
    P = Kp*err
    D = Kd*(err - err_arr[1])/delta_t
    I = Ki*np.sum(err_arr)*delta_t
    out = P + I + D


    if mode == "angle":
        if abs(out) > 25:
            out = np.sign(out)*25
            
            
        return int(out)
    else: 
        if abs(out) > 250:
            out = np.sign(out)*250  
        return int(out)