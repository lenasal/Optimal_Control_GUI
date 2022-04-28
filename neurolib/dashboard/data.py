import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path

from . import layout as layout
from . import functions as functions
from neurolib.utils import plotFunctions as plotFunc
from neurolib.utils import costFunctions as cost

background_color = layout.getcolors()[0]
cmap = layout.getcolormap()

background_dx_ = 0.025
background_dy_ = background_dx_

step_current_duration = layout.step_current_duration
max_step_current = layout.max_step_current

DC_duration = 100.

def set_parameters(model):
    model.params.sigma_ou = 0.
    model.params.mue_ext_mean = 0.
    model.params.mui_ext_mean = 0.
    model.params.ext_exc_current = 0.
    model.params.ext_inh_current = 0.
    
    # NO ADAPTATION
    model.params.IA_init = 0.0 * np.zeros((model.params.N, 1))  # pA
    model.params.a = 0.
    model.params.b = 0.
    
def remove_from_background(x, y, exc_1, inh_1):
    i = 0
    while i in range(len(x)):
        for j in range(len(exc_1)):
            if np.abs(x[i] - exc_1[j]) < 1e-4 and np.abs(y[i] - inh_1[j]) < 1e-4:
                x = np.delete(x, i)
                y = np.delete(y, i)
                break
        i += 1
    
    return x, y

def get_data_background(exc_1, inh_1, exc_2, inh_2):
    background_x, background_y = get_background(layout.x_plotrange[0], layout.x_plotrange[1], background_dx_,
                                                layout.y_plotrange[0], layout.y_plotrange[1], background_dy_)
    
    background_x, background_y = remove_from_background(background_x, background_y, exc_1, inh_1)
    background_x, background_y = remove_from_background(background_x, background_y, exc_2, inh_2)
    
    return go.Scatter(
    x=background_x,
    y=background_y,
    marker=dict(
        line=dict(
            width=2,
            color=background_color,
            ),
        color=background_color,
        size=[0] * len(background_x),
        symbol='x-thin',
    ),
    mode='markers',
    name='Background',
    hoverinfo='x+y',
    opacity=1.,
    showlegend=False,
    )

def get_background(xmin, xmax, dx, ymin, ymax, dy):
    x_range = np.arange(xmin,xmax+dx,dx)
    y_range = np.arange(ymin,ymax+dy,dy)

    n_x = len(x_range)
    n_y = len(y_range)

    background_x = np.zeros(( n_x * n_y ))
    background_y = background_x.copy()

    j_ = 0

    for x_ in x_range:
        for y_ in y_range:
            background_x[j_] = x_
            background_y[j_] = y_
            j_ += 1
            
    return background_x, background_y

def get_time(model, dur_):
    return np.arange(0., dur_/model.params.dt + model.params.dt, model.params.dt)
    
def get_step_current_traces(model):
    
    model.params.duration = step_current_duration
    stepcontrol_ = model.getZeroControl()
    stepcontrol_ = functions.step_control(model, maxI_ = max_step_current)
    time_ = get_time(model, step_current_duration)
    
    trace00 = go.Scatter(
        x=time_,
        y=stepcontrol_[0,0,:],
        xaxis="x",
        yaxis="y",
        name="External excitatory current [nA]",
        line_color=layout.darkgrey,
        showlegend=False,
        hoverinfo='x+y',
    )
    trace01 = go.Scatter(
        x=time_,
        y=stepcontrol_[0,1,:],
        xaxis="x",
        yaxis="y",
        name="External inhibitory current[nA]",
        line_color='rgba' + str(cmap(0)),
        showlegend=False,
        hoverinfo='x+y',
        visible=False,
    )
    return trace00, trace01

def trace_step(model, x_, y_):
    
    model.params.duration = step_current_duration
    model.params.mue_ext_mean = x_ * 5.
    model.params.mui_ext_mean = y_ * 5.
    stepcontrol_ = model.getZeroControl()
    stepcontrol_ = functions.step_control(model, maxI_ = max_step_current)
    
    time_ = get_time(model, step_current_duration)
    model.run(control=stepcontrol_)
    
    trace_exc = model.rates_exc[0,:] 
    trace_inh = model.rates_inh[0,:]
    
    return time_, trace_exc, trace_inh

def read_data_1(readpath, case):
    
    ind_, type_, mu_e, mu_i, cost_node = [], [], [], [], []
    # type: 0 - exc; 1 - inh; 2 - both; 3 - no solution; 4 - not checked
    
    file_ = os.sep + 'bi.pickle'
        
    if readpath[-1] == os.sep:
        readpath = readpath[:-1]
    
    if len(readpath) > 4:
        if readpath[-3] == '0':
            readpath_final = readpath[-2:]
            readpath = readpath[:-3]
            readpath = readpath + '1' + readpath_final    
    
    if not Path(readpath + file_).is_file():
        print("data not found")
        return [], [], [], [], [], [], [], [], [], [], []
    

    with open(readpath + file_,'rb') as file:
        load_array= pickle.load(file)
    ext_exc = load_array[0]
    ext_inh = load_array[1]
    
    ind_, type_, mu_e, mu_i = [None] * len(ext_exc), [None] * len(ext_exc), [None] * len(ext_exc), [None] * len(ext_exc)
    cost_node = [None] * len(ext_exc)

    [bestControl_0, bestState_0, costnode_0] = read_control(readpath, case)
    
    for i in range(len(ext_exc)):
        
        ind_[i] = i
        mu_e[i] = ext_exc[i]
        mu_i[i] = ext_inh[i]
        
        if type(bestControl_0[i]) is type(None):
            type_[i] = 6
            continue
        
        limit_signal = 1e-8
        max_signal = np.amax(np.abs(bestControl_0[i][0,:,:]))
                    
        if np.amax(np.abs(bestControl_0[i][0,:,:])) < limit_signal:
            type_[i] = 6
            bestControl_0[i] = None
            continue
        elif max_signal == np.amax(np.abs(bestControl_0[i][0,0,:])):
            type_[i] = 0
        elif max_signal == np.amax(np.abs(bestControl_0[i][0,1,:])):
            type_[i] = 1
        elif max_signal == np.amax(np.abs(bestControl_0[i][0,2,:])):
            type_[i] = 2
        elif max_signal == np.amax(np.abs(bestControl_0[i][0,3,:])):
            type_[i] = 3
        elif max_signal == np.amax(np.abs(bestControl_0[i][0,4,:])):
            type_[i] = 4
        elif max_signal == np.amax(np.abs(bestControl_0[i][0,5,:])):
            type_[i] = 5
        else:
            print(i, " no category")

            cost_node[i] = costnode_0[i]
    
    return ind_, type_, mu_e, mu_i, cost_node

def read_control(readpath, case):
    
    print('case = ', readpath, case)
    
    if readpath[-1] == os.sep:
        readpath = readpath[:-1]
    
    if case in ['1', '2', '3', '4']:
        readfile = readpath + os.sep + 'control_' + str(case) + '.pickle'
    elif case == '':
        readfile = readpath + os.sep + 'control.pickle'
    else:
        readfile = readpath + os.sep + 'control_' + str(case) + '.pickle'
    
    with open(readfile,'rb') as file:
        load_array = pickle.load(file)

    bestControl_ = load_array[0]
    bestState_ = load_array[1]
    costnode_ = load_array[3]
        
    return [bestControl_, bestState_, costnode_]

def get_scatter_data_1(type_, mu_e, mu_i):
    
    data0_x = []
    data0_y = []
    data1_x = []
    data1_y = []
    data6_x = []
    data6_y = []
    
    for i in range(len(type_)):
            if type_[i] == 0:
                data0_x.append(mu_e[i])
                data0_y.append(mu_i[i])
            elif type_[i] == 1:
                data1_x.append(mu_e[i])
                data1_y.append(mu_i[i])
            elif type_[i] == 6:
                data6_x.append(mu_e[i])
                data6_y.append(mu_i[i])


    data0 = go.Scatter(
        x=data0_x,
        y=data0_y,
        marker=dict(
            line=dict(width=1,
                      color='rgba' + str(cmap(3)),
                ),
            color='rgba' + str(cmap(3)),
            size=[layout.markersize] * len(data0_x)
        ),
        mode='markers',
        name='Excitatory control prevailing',
        #textfont=dict(size=layout.text_fontsize, color=layout.darkgrey),
        showlegend=True,
        hoverinfo='x+y',
        uid='1',
        )
        
    data1 = go.Scatter(
        x=data1_x,
        y=data1_y,
        marker=dict(
            line=dict(width=1,
                      color='rgba' + str(cmap(0)),
                ),
            color='rgba' + str(cmap(0)),
            size=[layout.markersize] * len(data1_x),
        ),
        mode='markers',
        name='Inhibitory control prevailing',
        hoverinfo='x+y',
        uid='2',
        )

    data6 = go.Scatter(
        x=data6_x,
        y=data6_y,
        marker=dict(
            line=dict(width=1,
                      color='rgba' + str(cmap(7)),
                ),
            color='rgba' + str(cmap(7)),
            size=[layout.markersize] * len(data6_x),
        ),
        mode='markers',
        name='No solution found',
        hoverinfo='x+y',
        uid='7',
        )

    return data0, data1, data6

def update_data(fig, be, bi, e1, i1, e2, i2):

    datab = fig.data[1]
    datab.x = e1
    datab.y = i1
    datab.marker.size=[0.] * len(be)
    if len(be) == 0:
        datab.x = [None]
        datab.y = [None]
    
    data1 = fig.data[1]
    data1.x = e1
    data1.y = i1
    data1.marker.size=[layout.markersize] * len(e1)
    if len(e1) == 0:
        data1.x = [None]
        data1.y = [None]
        
    data2 = fig.data[2]
    data2.x = e2
    data2.y = i2
    data2.marker.size=[layout.markersize] * len(e2)
    if len(e2) == 0:
        data2.x = [None]
        data2.y = [None]

def set_opt_cntrl_plot_zero(figure_, index_list):
    for i_ in index_list:
        figure_.data[i_].x = []
        figure_.data[i_].y = []
