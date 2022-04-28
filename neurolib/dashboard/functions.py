import numpy as np
import plotly.graph_objs as go
from . import layout as layout

def setmarkersize(index_, final_, trace_):
    s = list(trace_.marker.size)
    for ind_s in range(len(s)):
        if ind_s == index_:
            s[ind_s] = final_
    trace_.marker.size = s 
    
def setdefaultmarkersize(default_, trace_):
    s = list(trace_.marker.size)
    for ind_s in range(len(s)):
        s[ind_s] = default_
    trace_.marker.size = s 

def step_control(model, maxI_ = 1.):
    control_ = model.getZeroControl()
    for i_time in range(control_.shape[2]):
        if ( float(i_time/control_.shape[2]) < 0.1):
            control_[:,:1,i_time] = - maxI_
        elif ( float(i_time/control_.shape[2]) > 0.5 and float(i_time/control_.shape[2]) < 0.6 ):
            control_[:,:1,i_time] = maxI_
    return control_