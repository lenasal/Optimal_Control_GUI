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

def get_data_background(exc_1, inh_1, exc_2, inh_2, exc_4, inh_4):
    background_x, background_y = get_background(layout.x_plotrange[0], layout.x_plotrange[1], background_dx_,
                                                layout.y_plotrange[0], layout.y_plotrange[1], background_dy_)
    
    background_x, background_y = remove_from_background(background_x, background_y, exc_1, inh_1)
    background_x, background_y = remove_from_background(background_x, background_y, exc_2, inh_2)
    background_x, background_y = remove_from_background(background_x, background_y, exc_4, inh_4)
    
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

def plot_trace(model, x_, y_, trace0, trace1):
    model.params.duration = step_current_duration

    stepcontrol_ = model.getZeroControl()
    stepcontrol_ = functions.step_control(model, maxI_ = max_step_current)

    model.params.mue_ext_mean = x_ * 5.
    model.params.mui_ext_mean = y_ * 5.
    time_ = get_time(model, step_current_duration)

    model.run(control=stepcontrol_)
    
    trace0.x = time_
    trace0.y = model.rates_exc[0,:]
    
    trace1.x = time_
    trace1.y = model.rates_inh[0,:]
    
def setinit(model, init_vars_):
    init_vars = model.init_vars
    state_vars = model.state_vars
    for iv in range(len(init_vars)):
        for sv in range(len(state_vars)):
            if state_vars[sv] in init_vars[iv]:
                #print("set init vars ", )
                if model.params[init_vars[iv]].ndim == 2:
                    model.params[init_vars[iv]][0,:] = init_vars_[sv]
                else:
                    model.params[init_vars[iv]][0] = init_vars_[sv]
    
def DC_trace(model, x_, y_, start_, dur_, amp_, sim_dur, case_, trans_time_, weights,
             optimal_control, optimal_cost_node, optimal_weights, plot_ = False, max_it = 0):
    
    dt = model.params.dt

    model.params.mue_ext_mean = x_ * 5.
    model.params.mui_ext_mean = y_ * 5.
    
    model.params.duration = 3000.
    
    if case_ in ['1', '2']:
        maxI = 3.
    elif case_ in ['3', '4']:
        maxI = -3.
    elif case_[0] == '0':
        maxI = 3.
    else:
        maxI = -3.
            
    control0 = model.getZeroControl()
    control0 = functions.step_control(model, maxI_ = maxI)
    model.run(control=control0)

    target_rates = np.zeros((2))
    target_rates[0] = model.rates_exc[0,-1] 
    target_rates[1] = model.rates_inh[0,-1]

    control0 = functions.step_control(model, maxI_ = - maxI)
    model.run(control=control0)
        
    state_vars = model.state_vars

    init_state_vars = np.zeros(( len(state_vars) ))
    for j in range(len(state_vars)):
        if model.state[state_vars[j]].size == 1:
            init_state_vars[j] = model.state[state_vars[j]][0]
        else:
            init_state_vars[j] = model.state[state_vars[j]][0,-1]

    model.params.duration = sim_dur
    target_ = model.getZeroTarget()
    target_[:,0,:] = target_rates[0]
    target_[:,1,:] = target_rates[1]
        
    int_start = int( start_ / dt )
    int_stop = int_start + int( dur_ / dt )
    DC_control_ = model.getZeroControl()
    DC_control_[0,0,int_start:int_stop] = amp_[0]
    DC_control_[0,1,int_start:int_stop] = amp_[1]

    setinit(model, init_state_vars)
    model.run(control=DC_control_)
    state0_ = model.getZeroState()
    state0_[0,0,:] = model.rates_exc[0,:]
    state0_[0,1,:] = model.rates_inh[0,:]
    
    #print(state0_[0,0,0], state0_[0,0,-1], state0_[0,1,0], state0_[0,1,-1])
    #print(target_[0,0,0], target_[0,1,0])
        
    prec_variables = [0]
    
    T = int(sim_dur/dt + 1)
    target__ = target_.copy()
    for t in range(T):
        if t / T < trans_time_:
            target__[:,:,t] = -1000.
                
    cost_node = cost.cost_int_per_node(1, 6, int(sim_dur/dt + 1), dt, state0_, target__,
                                     DC_control_, weights[0], weights[1], weights[2], v_ = prec_variables )
        
    #print('precision cost: ', cost_node[0][0][0])
    #print('sparsity cost: ', cost_node[2][0][:])
    #print('energy cost: ', cost_node[1][0][:])
    
    if max_it == 0:
        if plot_:
            plotFunc.plot_control_current(model, [DC_control_, optimal_control],
                [cost_node, optimal_cost_node], [weights, optimal_weights, weights], sim_dur,
                0., 0., init_state_vars, target_, '', filename_ = '', transition_time_ = trans_time_,
                labels_ = ["DC control", "Optimal control"], print_cost_=False)
    
        return cost_node, DC_control_
    
    c_scheme = np.zeros(( 1,1 ))
    c_scheme[0,0] = 1.
    u_mat = np.identity(1)
    u_scheme = np.array([[1.]])
    cost.setParams(1.0, 0., 1.)
    
    setinit(model, init_state_vars)
        
    bestControl_, bestState_, cost_, runtime_, grad_, phi_, costnode_ = model.A1(
        DC_control_, target_, c_scheme, u_mat, u_scheme, max_iteration_ = max_it, tolerance_ = 1e-16,
        startStep_ = 10., max_control_ = np.array([5.,5.,0.,0.,0.,0.]), min_control_ = np.array([-5.,-5.,0.,0.,0.,0.]), t_sim_ = sim_dur,
        t_sim_pre_ = 10., t_sim_post_ = 10., CGVar = None, control_variables_ = [0],
        prec_variables_ = [0], transition_time_ = trans_time_)
    
    optimal_control_shift = np.zeros(( optimal_control.shape ))
    optimal_control_shift[:,:,:-1100] = optimal_control[:,:,1100:]
    
    setinit(model, init_state_vars)
    
    bestControl_shift, bestState_shift, cost_shift, runtime_shift, grad_shift, phi_shift, costnode_shift = model.A1(
        optimal_control_shift, target_, c_scheme, u_mat, u_scheme, max_iteration_ = max_it, tolerance_ = 1e-16,
        startStep_ = 10., max_control_ = np.array([5.,5.,0.,0.,0.,0.]), min_control_ = np.array([-5.,-5.,0.,0.,0.,0.]), t_sim_ = sim_dur,
        t_sim_pre_ = 10., t_sim_post_ = 10., CGVar = None, control_variables_ = [0],
        prec_variables_ = [0], transition_time_ = trans_time_)
        
    
    if plot_:
        plotFunc.plot_control_current(model, [DC_control_, optimal_control, bestControl_[:,:,100:-100], bestControl_shift[:,:,100:-100]],
            [cost_node, optimal_cost_node, costnode_, costnode_shift], [weights, optimal_weights, weights, weights], sim_dur,
            0., 0., init_state_vars, target_, '', filename_ = '', transition_time_ = trans_time_,
            labels_ = ["DC control", "Optimal control", "Optimal from DC", "Optimal shift"], print_cost_=False)
    
    return cost_node, DC_control_
    
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


def get_target(model, x_, y_, case_):
        
        dt = model.params.dt
        sim_duration = model.params.duration

        model.params.mue_ext_mean = x_ * 5.
        model.params.mui_ext_mean = y_ * 5.
    
        model.params.duration = 3000.
    
        if case_ in ['1', '2']:
            maxI = 3.
        elif case_ in ['3', '4']:
            maxI = -3.
        elif case_[0] == '0':
            maxI = 3.
        else:
            maxI = -3.
        control0 = model.getZeroControl()
        control0 = functions.step_control(model, maxI_ = maxI)
        
        model.run(control=control0)

        target_rates = np.zeros((2))
        target_rates[0] = model.rates_exc[0,-1] 
        target_rates[1] = model.rates_inh[0,-1]

        model.params.duration = sim_duration
        
        return target_rates

def read_data_1(model, readpath, case):
    
    ind_, type_, mu_e, mu_i, a_e, a_i, cost_node, w_e, w_i, target_high, target_low = [], [], [], [], [], [], [], [], [], [], []
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
    a_e, a_i, cost_node = [None] * len(ext_exc), [None] * len(ext_exc), [None] * len(ext_exc)
    w_e, w_i, target_high, target_low = [None] * len(ext_exc), [None] * len(ext_exc), [None] * len(ext_exc), [None] * len(ext_exc)

    [bestControl_init, costnode_init, bestControl_0, bestState_0, costnode_0] = read_control(readpath, case)
    
    for i in range(len(ext_exc)):
        
        ind_[i] = i
        mu_e[i] = ext_exc[i]
        mu_i[i] = ext_inh[i]
        
        if type(bestControl_0[i]) is type(None):
            type_[i] = 4
            continue
        
        cost_node[i] = costnode_0
        
        target_high[i] = get_target(model, ext_exc[i], ext_inh[i], '1')
        target_low[i] = get_target(model, ext_exc[i], ext_inh[i], '3')
                  
        target_rates = get_target(model, ext_exc[i], ext_inh[i], case)
                    
        if ( np.abs(np.mean(bestState_0[i][0,0,-50:]) - target_rates[0]) >
        0.3 * np.abs(np.mean(bestState_0[i][0,0,:50]) - target_rates[0])
        or np.abs(np.mean(bestState_0[i][0,1,-50:]) - target_rates[1]) >
        0.5 * np.abs(np.mean(bestState_0[i][0,1,:50]) - target_rates[1]) ):
            type_[i] = 3
            continue
            
        elif np.amax(np.abs(bestControl_0[i][0,1,:])) < 1e-8 and np.amax(np.abs(bestControl_0[i][0,0,:])) > 1e-8:
            type_[i] = 0
        elif np.amax(np.abs(bestControl_0[i][0,0,:])) < 1e-8 and np.amax(np.abs(bestControl_0[i][0,1,:])) > 1e-8:
            type_[i] = 1
        elif np.amax(np.abs(bestControl_0[i][0,0,:])) > 1e-8 and np.amax(np.abs(bestControl_0[i][0,1,:])) > 1e-8:
            type_[i] = 2
        elif np.amax(np.abs(bestControl_0[i][0,0,:])) < 1e-8 and np.amax(np.abs(bestControl_0[i][0,1,:])) < 1e-8:
            type_[i] = 3
            continue
        else:
            print(i, " no category")
    
        lenx = np.amax(bestControl_0[i][0,0,:])
        if np.abs(np.amin(bestControl_0[i][0,0,:])) > np.abs(lenx):
            lenx = np.amin(bestControl_0[i][0,0,:])
        leny = np.amax(bestControl_0[i][0,1,:])
        if np.abs(np.amin(bestControl_0[i][0,1,:])) > np.abs(leny):
            leny = np.amin(bestControl_0[i][0,1,:])
        a_e[i] = lenx
        a_i[i] = leny
        
        w_e[i] = get_width(bestControl_0[i][0,0,:], model.params.dt)
        w_i[i] = get_width(bestControl_0[i][0,1,:], model.params.dt)    
    
    return ind_, type_, mu_e, mu_i, a_e, a_i, cost_node, w_e, w_i, target_high, target_low

def get_width(node_control_, dt):
    start_ind = 0
    stop_ind = 0
    max_ = np.amax(np.abs(node_control_))
    for t in range(len(node_control_)):
        if start_ind == 0:
            if np.abs(node_control_[t]) >= max_/2.:
                start_ind = t
            continue
        else:
            if np.abs(node_control_[t]) < max_/2.:
                stop_ind = t
                break
            
    width = (stop_ind - start_ind) / dt
    return width
        

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
    costnode_ = load_array[6]
        
    return [None, None, bestControl_, bestState_, costnode_]

def get_scatter_data_1(ind_, type_, mu_e, mu_i, a_e, a_i):
    
    data1_x = []
    data1_y = []
    data2_x = []
    data2_y = []
    data4_x = []
    data4_y = []
    
    for i in range(len(type_)):
        if type_[i] in [0,1,2]:
            if np.abs(a_e[i]) > np.abs(a_i[i]):
                data1_x.append(mu_e[i])
                data1_y.append(mu_i[i])
            elif np.abs(a_e[i]) <= np.abs(a_i[i]):
                data2_x.append(mu_e[i])
                data2_y.append(mu_i[i])
        if type_[i] == 3:
            data4_x.append(mu_e[i])
            data4_y.append(mu_i[i])
    
    data1 = go.Scatter(
        x=data1_x,
        y=data1_y,
        marker=dict(
            line=dict(width=1,
                      color='rgba' + str(cmap(3)),
                ),
            color='rgba' + str(cmap(3)),
            size=[layout.markersize] * len(data1_x)
        ),
        mode='markers',
        name='Excitatory control prevailing',
        #textfont=dict(size=layout.text_fontsize, color=layout.darkgrey),
        showlegend=True,
        hoverinfo='x+y',
        uid='1',
        )

    data2 = go.Scatter(
        x=data2_x,
        y=data2_y,
        marker=dict(
            line=dict(width=1,
                      color='rgba' + str(cmap(0)),
                ),
            color='rgba' + str(cmap(0)),
            size=[layout.markersize] * len(data2_x),
        ),
        mode='markers',
        name='Inhibitory control prevailing',
        hoverinfo='x+y',
        uid='2',
        )
    
    data4 = go.Scatter(
        x=data4_x,
        y=data4_y,
        marker=dict(
            line=dict(width=1,
                      color='rgba' + str(cmap(7)),
                ),
            color='rgba' + str(cmap(7)),
            size=[layout.markersize] * len(data4_x),
        ),
        mode='markers',
        name='No solution found',
        hoverinfo='x+y',
        uid='4',
        )

    return data1, data2, data4
    

def get_scatter_data(exc_1, inh_1, exc_2, inh_2, exc_3, inh_3, exc_4, inh_4):

    data1 = go.Scatter(
        x=exc_1,
        y=inh_1,
        marker=dict(
            line=dict(width=1,
                      color='rgba' + str(cmap(3)),
                ),
            color='rgba' + str(cmap(3)),
            size=[layout.markersize] * len(exc_1)
        ),
        mode='markers',
        name='Excitatory current only',
        hoverinfo='x+y',
        uid='1',
        )
    
    if len(exc_1) == 0:
        data1.x = [None]
        data1.y = [None]

    data2 = go.Scatter(
        x=exc_2,
        y=inh_2,
        marker=dict(
            line=dict(width=1,
                      color='rgba' + str(cmap(0)),
                ),
            color='rgba' + str(cmap(0)),
            size=[layout.markersize] * len(exc_2),
        ),
        mode='markers',
        name='Inhibitory current only',
        hoverinfo='x+y',
        uid='2',
        )
    
    if len(exc_2) == 0:
        data2.x = [None]
        data2.y = [None]
    
    
    data3 = go.Scatter(
        x=exc_3,
        y=inh_3,
        marker=dict(
            line=dict(width=1,
                      color='rgba' + str(cmap(2)),
                ),
            color='rgba' + str(cmap(2)),
            size=[layout.markersize] * len(exc_3)
        ),
        mode='markers',
        name='Control in both nodes',
        hoverinfo='x+y',
        uid='3',
        )
    
    if len(exc_3) == 0:
        data3.x = [None]
        data3.y = [None]

    data4 = go.Scatter(
        x=exc_4,
        y=inh_4,
        marker=dict(
            line=dict(width=1,
                      color='rgba' + str(cmap(7)),
                ),
            color='rgba' + str(cmap(7)),
            size=[layout.markersize] * len(exc_4),
        ),
        mode='markers',
        name='No control result',
        hoverinfo='x+y',
        uid='4',
        )
    
    if len(exc_4) == 0:
        data4.x = [None]
        data4.y = [None]
    
    return data1, data2, data3, data4

def update_data(fig, e1, i1, e2, i2, e3, i3, e4, i4):
    
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
        
    data3 = fig.data[3]
    data3.x = e3
    data3.y = i3
    data3.marker.size=[layout.markersize] * len(e3)
    if len(e3) == 0:
        data3.x = [None]
        data3.y = [None]
        
    data4 = fig.data[4]
    data4.x = e4
    data4.y = i4
    data4.marker.size=[layout.markersize] * len(e4)
    if len(e4) == 0:
        data4.x = [None]
        data4.y = [None]
        
def dist_right(e_, i_, exc__, inh__, grid_resolution_):
    row = []
    for i in range(len(inh__)):
        if np.abs(i_ - inh__[i]) < 1e-6:
            row.append(exc__[i])
    upper_bound = max(row)
    dist = upper_bound - e_ + grid_resolution_/2.
    return dist

def dist_left(e_, i_, exc__, inh__, grid_resolution_):
    row = []
    for i in range(len(inh__)):
        if np.abs(i_ - inh__[i]) < 1e-6:
            row.append(exc__[i])
    lower_bound = min(row)
    dist = e_ - lower_bound + grid_resolution_/2.
    return dist

def dist_low(e_, i_, exc__, inh__, grid_resolution_):
    column = []
    for i in range(len(exc__)):
        if np.abs(e_ - exc__[i]) < 1e-6:
            column.append(inh__[i])
    lower_bound = min(column)
    dist = i_ - lower_bound + grid_resolution_/2.
    return dist

def dist_up(e_, i_, exc__, inh__, grid_resolution_):
    column = []
    for i in range(len(inh__)):
        if np.abs(e_ - exc__[i]) < 1e-6:
            column.append(inh__[i])
    upper_bound = max(column)
    dist = upper_bound - i_ + grid_resolution_/2.
    return dist

def set_opt_cntrl_plot_zero(figure_, index_list):
    for i_ in index_list:
        figure_.data[i_].x = []
        figure_.data[i_].y = []
    
def set_data(fig, index, data):
    fig.data[index].x = data.x
    fig.data[index].y = data.y