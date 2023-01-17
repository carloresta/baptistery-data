#===============
#   IMPORTS
#===============

# Fundamentals
import numpy as np
import pandas as pd

# Dash
import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#import plotly.io as pio
#from plotly.tools import mpl_to_plotly

#-STYLES----------------------------------
linestyles = ['-', '--', '-.', ':', (0, (5, 2, 1, 2, 1, 2))]
markers = ['o', 'x', 's', '^', 'd']
colors = ['#0077BB', '#33BBEE', '#009988', '#EE7733', '#CC3311', '#EE3377', '#BBBBBB']
colors4 = ['#0077BB', '#EE7733', '#CC3311', '#EE3377']
blues = ['#eff3ff','#c6dbef','#9ecae1','#6baed6','#3182bd','#08519c']
greys5 = ['#f7f7f7','#cccccc','#969696','#636363','#252525']
greys6 = ['#f7f7f7','#d9d9d9','#bdbdbd','#969696','#636363','#252525']
greys7 = ['#f7f7f7','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525']
greys8 = ['#ffffff','#f0f0f0','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525']
greys9 = ['#ffffff','#f0f0f0','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525','#000000']
hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

#====================
#   PY INCLUSIONS
#====================
# Data import
import baptistery_data_input as bdi
prism_data, levelling_data, extensimeter_data = bdi.readAllData()
prism_pos, levelling_pos, extensimeter_pos, positions = bdi.readSensorPositions()


#=====================
#   MISC FUNCTIONS
#=====================
def scaleFactorCalc(exp, factor):
    return factor*10**exp


def xyToEN(x, y):
    """
    Converts (x, y) coordinates of prisms to (E, N) ones.
    It takes either single or array-like coordinates.
    """
    ref_x, ref_y = 15.184322095298622, -0.01676310147012092
    rot = np.deg2rad(37.1)
    e = x - ref_x
    n = y - ref_y
    eR = e*np.cos(rot) - n*np.sin(rot)
    nR = e*np.sin(rot) + n*np.cos(rot)
   
    return eR, nR


def interpolateRGB(start, end, n):
    """
    Returns *n* RGB color tuples interpolating
    linearly from start to end. 
    Start and end must be list-like with rgb values.
    """
    fractions = np.linspace(0, 1, num=n)
    r = (end[0] - start[0])*fractions+start[0]
    g = (end[1] - start[1])*fractions+start[1]
    b = (end[2] - start[2])*fractions+start[2]
    colors = ['rgb({:0f},{:0f},{:0f})'.format(r[i], g[i], b[i]) for i in range(len(fractions))]
    return colors


def selectPrismSection(n):
    """
    NOTE: for now, the function excludes prisms in the 
    1xx series.    
    """
    full_section = {
        '01':['01','07'],
        '02':['02','08'],
        '03':['03','09'],
        '04':['04','10'],
        '05':['05','11'],
        '06':['06','12'],
        '07':['07','01'],
        '08':['08','02'],
        '09':['09','03'],
        '10':['10','04'],
        '11':['11','05'],
        '12':['12','06']
    }
    n = full_section[n]
   
    selected_sensors = [p for p in prism_pos.index if p.endswith(n[0]) and not (p.startswith('1'))]
    selected_sensors += [p for p in prism_pos.index if p.endswith(n[1]) and not (p.startswith('1'))]
    
    return selected_sensors    


def rotTraslPrism(df, date):
    """
    Returns east and z coordinates of prisms
    in the DataFrame *df* for the date *date*.
    """
    x = df.loc[date, (slice(None), 'x')].values
    y = df.loc[date, (slice(None), 'y')].values
    z = df.loc[date, (slice(None), 'z')].values
    
    # Traslation
    ref_x, ref_y = 15.184322095298622, -0.01676310147012092
    x1 = x - ref_x
    y1 = y - ref_y
    
    # Rotation
    a = -np.arctan(y1/x1)
    east = x1*np.cos(a) - y1*np.sin(a)
    north = x1*np.sin(a) + y1*np.cos(a)

    if (north > 10**(-6)).any():
        print("ERROR: north coordinate is not zero")
        return 1
    
    return east, z

#=====================
#   PLOT FUNCTIONS
#=====================
def reformatPlot(fig, size=None, secondary=False):
    """
    Updates the layout of all subplots.
    """
    if size != None:
        fig.update_layout(height=size[1], width=size[0])
        
    fig.update_xaxes(
        showline = True,
        mirror = 'ticks',
        linewidth = 2,
        ticks = 'inside',
        ticklen = 8,
        tickwidth = 2,
        showgrid = True,
        title_font_size=16,
        tickfont_size=16,
    )
    fig.update_yaxes(
        showline = True,
        mirror = 'ticks',
        linewidth = 2,
        ticks = 'inside',
        ticklen = 8,
        tickwidth = 2,
        showgrid = True,
        title_font_size=16,
        tickfont_size=16,
    )
    
    if secondary:
        fig.update_layout(
                yaxis2=dict(
                title_font_size=16,
                tickfont_size=16,
                showgrid=False,
                showline=True,
                linewidth=2,
                ticks='inside',
                ticklen=8,
                tickwidth=2,
                title_font_color='gray',
                tickcolor='lightgray',
                tickfont_color='gray'
            ),
            yaxis_mirror=False
        )
        
    return fig 


def figureGantt(which):
    """
    Plots a Gantt chart with the temporal availability of data.
    Expects:
    - which = list of instrumentation to plot.
    Returns:
    - figure object
    """
    which_df = {'Prisms': prism_data,
               'Levelling': levelling_data,
               'Cracks': extensimeter_data}

    fig = go.Figure(layout_template=None)
    
    for i, w in enumerate(which):
        df = which_df[w]
        my_index = df[~df.index.duplicated()].resample('D').ffill().index
        fig.add_trace(
            go.Scatter(x=my_index, y=[w]*len(my_index),
                   mode='markers', name=w, 
                      marker_color=colors[i],
                      marker_size = 10,
                      marker_symbol='square')
        )
    
    fig = reformatPlot(fig, [750,400])
    fig.update_layout(dict(yaxis_range=(-0.5, len(which)-0.5)),
                     margin=dict(r=20, t=20))
    fig.update_layout(dict(showlegend=False))
    fig.update_layout(hovermode='x unified')
    fig.update_traces(hovertemplate="%{x}")
    return fig

def figurePrismPlan(daterange, scalefactor, floor):
    """
    Produces a plot showing the displacement of prisms
    in plan.
    Expects:
    - daterange: a list of two integers within len(prism_data.index)
    - scalefactor: a number
    - floor: a list of strings, either/or "First" and "Second"  
    Returns:
    - the plot
    """
    fig = go.Figure(layout_template=None)
    fig = reformatPlot(fig, size=[800,700])

    # Select prisms
    if floor == []:
        return fig
    if 'First' in floor:
        selected_prisms = [i for i in prism_pos.index if (x:=i[0]) == '2' or x=='3']
        if 'Second' in floor:
            selected_prisms += [i for i in prism_pos.index if (x:=i[0]) == '4' or x=='5']
    else:
        selected_prisms = [i for i in prism_pos.index if (x:=i[0]) == '4' or x=='5']
    df = prism_data.loc[:, (selected_prisms, slice(None))]
    
    # Get prism coordinates
    start_date = df.index[daterange[0]]
    end_date = df.index[daterange[1]]
    x_0 = df.loc[start_date, (slice(None), 'x')].values
    y_0 = df.loc[start_date, (slice(None), 'y')].values
    x_1 = df.loc[end_date, (slice(None), 'x')].values
    y_1 = df.loc[end_date, (slice(None), 'y')].values
    
    # Rotate prisms so that x is in the East direction
    # and translate them so that the origin is in the center
    # of the Baptistery
    eR_0, nR_0 = xyToEN(x_0, y_0)
    eR_1, nR_1 = xyToEN(x_1, y_1)
    east_diff = (eR_1 - eR_0) * scalefactor
    north_diff = (nR_1 - nR_0) * scalefactor
    eR_1 = eR_0 + east_diff
    nR_1 = nR_0 + north_diff
    
    # Plot prisms
    fig.add_trace(
        go.Scatter(x=eR_0, y=nR_0,
                   mode='markers', name=str(start_date)[:10],
                   marker_size = 10,
                   marker_color = blues[1],
                   hovertext = ["Prism n. {}".format(i) for i in selected_prisms],
                   hoverinfo = 'text'
        )
    )
    fig.add_trace(
        go.Scatter(x=eR_1, y=nR_1,
                   mode='markers', name=str(end_date)[:10],
                   marker_size = 10,
                   marker_color = blues[-2],
                   hovertext = ["Prism n. {}".format(i) for i in selected_prisms],
                   hoverinfo = 'text'
        )
    )
    
    # Plot lines connecting corresponding points
    for i in range(len(eR_0)):
        fig.add_shape(type="line",
            x0=eR_0[i], y0=nR_0[i], 
            x1=eR_1[i], y1=nR_1[i],
            line=dict(
                color="lightgrey",
                width=2,
                dash="dot"
            ))
    
    # Add the shape of the Baptistery
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=-17.80, y0=-17.80, x1=17.80, y1=17.80,
        line_color="lightgrey"
    )
                      
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=-15.25, y0=-15.25, x1=15.25, y1=15.25,
        line_color="lightgrey"
    )
    
    # Format plot
    fig.update_layout(dict(
        xaxis_range = (-25, 25),
        yaxis_range = (-25, 25),
    ),
        margin=dict(t=40)
                     )
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
      )
    
    return fig


def figurePrismSection(selected_prisms, daterange, scalefactor, fixedbase):
    """
    Produces a plot showing the displacement of prisms in a given section.
    Expects:
    - selected_prisms: a list of prism names
    - daterange: a list of two integers within len(prism_data.index)
    - scalefactor: a number
    - fixedbase: boolean  
    Returns:
    - the plot    
    """
        
    # This part assumes that the prisms in each
    # section use the same naming convention
    selected_prisms.sort()    
    links = [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7]
    ]
    
    fig = go.Figure(layout_template = None)
    fig = reformatPlot(fig, size=[800, 700])
    fig.update_layout(
        margin=dict(t=40)
    )
    
    # Prune dataframe
    df = prism_data.loc[:, (selected_prisms, slice(None))]
    # Selection dates
    numdate = daterange[1] - daterange[0] + 1
    dates = [df.index[daterange[0]], df.index[daterange[1]]]
    # Colors
    colors = ['rgb(255, 198, 196)', 'rgb(103, 32, 68)']
    
    e0, z0 = rotTraslPrism(df, dates[0])
    fig.add_trace(
        go.Scatter(
            x=e0, y=z0,
            name=(ddd:=str(dates[0])[:10]),
            mode='markers', marker_size=10,
            marker_color=colors[0],
            hovertext=['Prism n. {}\n{}'.format(i, ddd)
                          for i in selected_prisms],
                hoverinfo='text'
        )
    )    
    for l in links:
        fig.add_shape(type="line",
            x0=e0[l[0]], y0=z0[l[0]], 
            x1=e0[l[1]], y1=z0[l[1]],
            line=dict(
                color=colors[0],
                width=2,
            ))
        
        if fixedbase:
            fig.add_shape(type='line',
                     x0=e0[l[0]], y0=z0[l[0]],
                     x1=e0[l[0]], y1=0.0,
                     line=dict(
                         color=colors[0],
                         width=1,
                         dash='dash'
                     ))
    
    for i, d in enumerate(dates[1:]):
        e, z  = rotTraslPrism(df, d)
        diff_e = e-e0
        diff_z = z-z0
        e = e0 + diff_e * scalefactor
        z = z0 + diff_z * scalefactor
        fig.add_trace(
            go.Scatter(
                x = e, y = z,
                mode = 'markers', marker_size=10,
                name = (ddd:=str(d)[:10]),
                marker_color=colors[i+1],
                hovertext=['Prism n. {}\n{}'.format(i,ddd)
                          for i in selected_prisms],
                hoverinfo='text'
            )
        )
        for l in links:
            fig.add_shape(type="line",
                x0=e[l[0]], y0=z[l[0]], 
                x1=e[l[1]], y1=z[l[1]],
                line=dict(
                    color=colors[i+1],
                    width=1
                ))
            
            if fixedbase:
                fig.add_shape(type='line',
                             x0=e[l[0]], y0=z[l[0]],
                             x1=e0[l[0]], y1=0.0,
                             line=dict(
                                 color=colors[i+1],
                                 width=1,
                                 dash='dash'
                             ))
    return fig



def figureSectionSelection(selected_prisms):
    """
    A small figure showing which section was selected.
    """
    fig = go.Figure(layout_template='plotly_white')
    fig.update_layout(height=250, width=250,
                         margin=dict(l=0,r=0,b=0,t=0))
    
    unselected_prisms = [el for el in prism_pos.index if (el not in selected_prisms and el[0] != '1')]
    up_x = [(a:=prism_pos.loc[i])['radius'] * np.cos(np.deg2rad(a['angle'])) 
            for i in unselected_prisms] 
    up_y = [(a:=prism_pos.loc[i])['radius'] * np.sin(np.deg2rad(a['angle'])) 
            for i in unselected_prisms]
    sp_x = [(a:=prism_pos.loc[i])['radius'] * np.cos(np.deg2rad(a['angle']))
            for i in selected_prisms] 
    sp_y = [(a:=prism_pos.loc[i])['radius'] * np.sin(np.deg2rad(a['angle']))
            for i in selected_prisms]
        
    # Add the shape of the Baptistery
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=-17.80, y0=-17.80, x1=17.80, y1=17.80,
        line_color="lightgrey"
    )                
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=-15.25, y0=-15.25, x1=15.25, y1=15.25,
        line_color="lightgrey",
        line_width=1
    )
    
    # Add unselected prisms
    fig.add_trace(
        go.Scatter(x=up_x, y=up_y,
                  mode='markers',
                  marker_color='lightblue',
                  hovertext=['Prism n. {}'.format(i) 
                             for i in unselected_prisms],
                  hoverinfo='text'
                  )
    )
    
    # Add selected prisms
    fig.add_trace(
        go.Scatter(x=sp_x, y=sp_y,
                  mode='markers',
                  marker_color='red',
                  hovertext=['Prism n. {}'.format(i) 
                             for i in selected_prisms],
                  hoverinfo='text'
                  )
    )
    
    # Disable the legend
    fig.update(layout_showlegend=False)
    # Format plot
    fig.update_layout(dict(
        xaxis_range = (-20, 20),
        yaxis_range = (-20, 20),
        xaxis_zeroline = False,
        xaxis_showgrid = False,
        yaxis_zeroline = False,
        yaxis_showgrid = False,
        xaxis_visible = False,
        yaxis_visible = False
    ))
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
      )
    
    return fig  


def figurePrismDisplacement(p):
    """
    Produces a figure with the displacement of the *p*
    prism. Both the total one and single components.
    """
    
    fig = go.Figure(layout_template=None)
    fig.update_layout(
        margin = dict(t=40, b=40),
    )
    
    fig = make_subplots(specs=[[{"secondary_y": True}]],
                        shared_xaxes=True,
                        vertical_spacing=0.0,
                        figure=fig)
    fig = reformatPlot(fig, size=[1200, 350], secondary=True)
    
    x = prism_data[p, 'x'].values
    y = prism_data[p, 'y'].values
    z = prism_data[p, 'z'].values*1000.
    e, n = xyToEN(x, y)
    e = e*1000.
    n = n*1000.

    de = e - e[0]
    dn = n - n[0]
    dz = z - z[0]

    dtot = np.sqrt(de**2 + dn**2 + dz**2)

    r = np.sqrt(e**2 + n**2)
    dr = r - r[0]

    alpha = np.arctan(n/e)
    dalpha = alpha - alpha[0]
    dtan = r*np.sin(dalpha)

    traces = [dtot, dr, dtan, dz]
    names = ['Total', 'Radial', 'Tangential', 'Vertical']

    for t,n,c in zip(traces, names, colors4):
        fig.add_trace(
            go.Scatter(
                x=prism_data.index,
                y=t,
                mode='markers+lines',
                name=n,
                marker_color = c,
                line_color = c,
            ),
            secondary_y = False,
        )

    fig.add_trace(
        go.Scatter(
            x=extensimeter_data.index,
            y=extensimeter_data['F4F8', 'temp'].rolling(24).mean(),
            line_dash='dot',
            line_color='gray',
            name='Temperature'
        ),
        secondary_y = True,
    )

    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.02,
        y=0.95,
        text='<b>'+p+'</b>',
        font_family='Roboto',
        font_color='black',
        font_size=14,
        borderpad=4,
        bordercolor='black',
        borderwidth=1.5,
        showarrow=False
    )

    fig.update_yaxes(title_text="Displacement [mm]", secondary_y=False)
    fig.update_yaxes(title_text="Temperature [°C]", secondary_y=True)
            
    return fig


#=========================
#   STANDALONE FIGURES
#=========================
def figurePrismSelection():
    """
    A figure from which to select prisms.
    """
    fig = go.Figure(layout_template='plotly_white')
    
    fig = make_subplots(rows=1, cols=3, 
                        subplot_titles=('Ground floor', 'First floor', 'Second floor'),
                        figure=fig
                       )
    fig.update_layout(height=350, width=1000,
                         margin=dict(l=0,r=0,b=0,t=60))
    
    ground_floor = ['101', '102', '103', '104', 'P1']
    first_floor = [p for p in prism_pos.index if (p[0] == '2' or p[0] == '3')]
    second_floor = [p for p in prism_pos.index if (p[0] == '4' or p[0] == '5')]
    
    floors = [ground_floor, first_floor, second_floor]
    
    for i, f in enumerate(floors):
        x = prism_pos.loc[f]['radius']*np.cos(np.deg2rad(prism_pos.loc[f]['angle']))
        y = prism_pos.loc[f]['radius']*np.sin(np.deg2rad(prism_pos.loc[f]['angle']))
        # Add the shape of the Baptistery
        fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=-17.80, y0=-17.80, x1=17.80, y1=17.80,
            line_color="lightgrey",
            row=1, col=i+1
        )                
        fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=-15.25, y0=-15.25, x1=15.25, y1=15.25,
            line_color="lightgrey",
            line_width=1,
            row=1, col=i+1
        )
        # Add the prisms
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                customdata=f,
                text=f,
                textfont_size=10,
                textposition='top center',
                mode='markers+text',
                marker_color=colors[1],
                hovertext=['Prism n. {}'.format(p) 
                             for p in f],
                marker_size=10,
                hoverinfo='text',
                selected_marker_color='red'
              ),
            row=1, col=i+1
        )
        
    # Disable the legend
    fig.update(layout_showlegend=False)
    
    # Format plot
    fig.update_xaxes(
        range = (-18, 18),
        zeroline = False,
        showgrid = False,
        visible = False,
    )
    fig.update_yaxes(
        range = (-18, 18),
        zeroline = False,
        showgrid = False,
        visible = False,
    )
    fig.update_layout(
        yaxis1=dict(
        scaleanchor = "x1",
        scaleratio = 1,
        )
    )
    fig.update_layout(
        yaxis2=dict(
        scaleanchor = "x2",
        scaleratio = 1,
        )
    )
    fig.update_layout(
        yaxis3=dict(
        scaleanchor = "x3",
        scaleratio = 1,
        )
    )
    
    fig.update_layout(clickmode='event+select')
    
    return fig

fig_prism_selection = figurePrismSelection()


def figureSectionRelativeDisplacements(prisms):
    """
    Produces a plot with the relative displacements of
    corresponding prisms in a section.
    """
    fig = go.Figure(layout_template=None)
    fig.update_layout(margin = dict(t=40, b=40))
    fig = make_subplots(specs=[[{"secondary_y": True}]],
                        figure=fig)
    fig = reformatPlot(fig, size=[900, 350], secondary=True)
    
    dists = np.sqrt(((prism_data[prisms[1]] - prism_data[prisms[0]])**2).sum(axis=1))*1000
    rel_disp = dists - dists[0]
    
    fig.add_trace(
        go.Scatter(
            x=prism_data.index,
            y=rel_disp,
            mode='markers+lines',
            name='Relative displacement',
            marker_color='#009988',
            line_color='#009988'
        ),
        secondary_y=False,
    )  
    
    fig.add_trace(
        go.Scatter(
            x=extensimeter_data.index,
            y=extensimeter_data['F4F8', 'temp'].rolling(24).mean(),
            line_dash='dot',
            line_color='gray',
            name='Temperature'
        ),
        secondary_y = True,
    )    
    
    fig.add_annotation(
        xref="x domain",
        yref="y domain",
        x=0.02,
        y=0.95,
        text='<b>'+prisms[0] + '-' + prisms[1]+'</b>',
        font_family='Roboto',
        font_color='black',
        font_size=14,
        borderpad=4,
        bordercolor='black',
        borderwidth=1.5,
        showarrow=False
    )
    
    fig.update_yaxes(title_text="Displacement [mm]", secondary_y=False)
    fig.update_yaxes(title_text="Temperature [°C]", secondary_y=True)
    
    fig.update_layout(
        legend=dict(
            x=0.70, y=0.95
        )
    )
    
    return fig

def figurePrismCoupleSelection(selected_prisms):
    """
    A small figure showing which couple of prisms was selected.
    """
    fig = go.Figure(layout_template='plotly_white')
    fig.update_layout(height=250, width=250,
                         margin=dict(l=0,r=0,b=0,t=0))
    
    unselected_prisms = [el for el in prism_pos.index if (el not in selected_prisms and el[0] != '1')]
    up_x = [(a:=prism_pos.loc[i])['radius'] * np.cos(np.deg2rad(a['angle'])) 
            for i in unselected_prisms] 
    up_y = [(a:=prism_pos.loc[i])['radius'] * np.sin(np.deg2rad(a['angle'])) 
            for i in unselected_prisms]
    sp_x = [(a:=prism_pos.loc[i])['radius'] * np.cos(np.deg2rad(a['angle']))
            for i in selected_prisms] 
    sp_y = [(a:=prism_pos.loc[i])['radius'] * np.sin(np.deg2rad(a['angle']))
            for i in selected_prisms]
        
    # Add the shape of the Baptistery
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=-17.80, y0=-17.80, x1=17.80, y1=17.80,
        line_color="lightgrey"
    )                
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=-15.25, y0=-15.25, x1=15.25, y1=15.25,
        line_color="lightgrey",
        line_width=1
    )
    
    # Add unselected prisms
    fig.add_trace(
        go.Scatter(x=up_x, y=up_y,
                  mode='markers',
                  marker_color='lightblue',
                  hovertext=['Prism n. {}'.format(i) 
                             for i in unselected_prisms],
                  hoverinfo='text'
                  )
    )
    
    # Add selected prisms
    fig.add_trace(
        go.Scatter(x=sp_x, y=sp_y,
                  mode='markers+text',
                  text=selected_prisms,
                  textfont_family='Roboto',
                  textposition='top center',
                  textfont_size=12,
                  marker_color='red',
                  hovertext=['Prism n. {}'.format(i) 
                             for i in selected_prisms],
                  hoverinfo='text'
                  )
    )
    
    # Connecting line
    fig.add_shape(type="line",
            x0=sp_x[0], y0=sp_y[0], 
            x1=sp_x[1], y1=sp_y[1],
            line=dict(
                color='gray',
                width=2,
                dash='dash'
            ))
    
    # Disable the legend
    fig.update(layout_showlegend=False)
    # Format plot
    fig.update_layout(dict(
        xaxis_range = (-20, 20),
        yaxis_range = (-20, 20),
        xaxis_zeroline = False,
        xaxis_showgrid = False,
        yaxis_zeroline = False,
        yaxis_showgrid = False,
        xaxis_visible = False,
        yaxis_visible = False
    ))
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
      )
    
    return fig  




#=================
#   APP SETUP
#=================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])


#======================
#     APP LAYOUT
#======================
# The webapp will be organized into tabs, each one containing specific
# kind of info or graphics. To keep the code tidy, each tab is defined
# separately, and then all are collected into the "Outer level" layout.

#-----------------
#      TABS
#-----------------

# INFO Tab
#----This tab contains info regarding the available 
#----monitoring data.
tab_info = dbc.Tab([
    html.Br(),
    html.P("This tab contains information regarding sensors installed on the Baptistery and available data.",
          style=dict(color='grey', fontSize='small')),
    html.H2('Installed instrumentation'),
    html.P("""
        Lorem ipsum...
    """),
    html.H2('Temporal availability of data'),
    html.P("""
        The following plot shows data availability in time.
    """),
    dbc.Row([
        dbc.Col(
            dcc.Graph(id='fig_gantt',
                     config=dict(
                         displayModeBar=False
                     )),
            width=9,
            align='center'
        ),
        dbc.Col([
            dcc.Markdown("**Sensors to show**"),
            dbc.Checklist(
                id='checklist_gantt',
                options=['Prisms', 'Levelling', 'Cracks'],
                value=['Prisms', 'Levelling', 'Cracks'],
                inline=False
                )
        ],
            width=3,
            align='center'
        )
    ], justify='center')
], label='INFO')


# CHECKS Tab
#----This tab will contain data checks:
#------1) comparison of vertical prisms and levelling
#------2) ???
tab_checks = dbc.Tab([
    html.Div([
        html.Br(),
        html.P('To be completed... 🚧')
    ])
], label='CHECKS')


# PLAN Tab
#----This tab will contain plan plots.
tab_plan = dbc.Tab([
    html.Div([
        html.Br(),
        html.P("This tab contains plots of prisms in plan view.",
          style=dict(color='grey', fontSize='small')),
    ]),
    html.Div([
        html.H2("Plan view of prisms"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id='fig_prism_plan'),
            width=9, align='center'
            ),
            dbc.Col([
                dcc.Markdown("**Date range**"),
                dcc.RangeSlider(id='slider_prism_plan_daterange',
                               min=0, max=(x:=len(prism_data)-1), step=1,
                               value=[0, x],
                               marks=None,
                               allowCross=False,
                               pushable=True,
                               updatemode='drag'
                ),
                dcc.Markdown(id='text_prism_plan_daterange'),
                html.Br(),
                dcc.Markdown('**Scaling factor**'),
                dcc.Slider(id='slider_prism_plan_scalefactor_log',
                          min=0, max=3, step=1,
                          marks={i: '{}'.format(10 ** i) for i in range(4)},
                          value=3,
                          updatemode='drag'),
                dcc.Slider(id='slider_prism_plan_scalefactor_dec',
                           min=0, max=10, step=1,
                           marks={i: '{}'.format(i) for i in [0, 2, 4, 6, 8, 10]},
                           value=3,
                           updatemode='drag'),
                dcc.Markdown(id='text_prism_plan_scalefactor'),
                html.Br(),
                dcc.Markdown('**Floor selection**'),
                dbc.Checklist(id='checklist_prism_plan_floor',
                              options=['First', 'Second'],
                              value=['Second'],
                              inline=False
                             )
            ],
            width=3, align='center'
            )
        ])
    ])
], label='PLAN')


# SECTION Tab
#----This tab will contain section plots.
tab_section = dbc.Tab([
    html.Div([
        html.Br(),
        html.P("This tab contains plots of prisms in section view.",
          style=dict(color='grey', fontSize='small'))
    ]),
    html.Div([
        html.H2("Section view of the prisms"),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id='fig_prism_section'),
            width=9, align='center'
            ),
            dbc.Col([
                dcc.Markdown('**Select a section**'),
                dcc.Slider(id='slider_prism_section_selection',
                          min=1, max=12, step=1,
                          value=5, 
                          #marks=None,
                          #tooltip={"placement": "bottom", "always_visible": True},
                           updatemode='drag'
                          ),
                dcc.Graph(id='fig_prism_section_selection',
                         config=dict(
                         displayModeBar=False,
                     )),
                html.Br(),
                dcc.Markdown("**Date range**"),
                dcc.RangeSlider(id='slider_prism_section_daterange',
                               min=0, max=(x:=len(prism_data)-1), step=1,
                               value=[0, x],
                               marks=None,
                               allowCross=False,
                               pushable=True,
                               updatemode='drag'
                ),
                dcc.Markdown(id='text_prism_section_daterange'),
                html.Br(),
                dcc.Markdown('**Scaling factor**'),
                dcc.Slider(id='slider_prism_section_scalefactor_log',
                          min=0, max=3, step=1,
                          marks={i: '{}'.format(10 ** i) for i in range(4)},
                          value=3,
                          updatemode='drag'),
                dcc.Slider(id='slider_prism_section_scalefactor_dec',
                           min=0, max=10, step=1,
                           marks={i: '{}'.format(i) for i in [0, 2, 4, 6, 8, 10]},
                           value=3,
                           updatemode='drag'),
                dcc.Markdown(id='text_prism_section_scalefactor'),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dcc.Markdown('**Fixed base?**'),
                    ]),
                    dbc.Col([
                        dbc.Checklist(id='checklist_prism_section_fixedbase',
                                  options=[{'label':'', 'value':1}],
                                  value=[1],
                                  inline=True,
                                  switch=True
                                 )
                    ])
                ])
            ],
            width=3, align='center'
            )
        ])
    ]),
    html.Br(),
    html.Div([
        html.H2('Relative displacement of selected prisms'),
        html.Div(id='div_relative_displacement_plots')
    ])
    
], label='SECTION')



# PRISMS Tab
#----This tab will contain section plots.
tab_prisms = dbc.Tab([
    html.Div([
        html.Br(),
        html.P("This tab contains plots of prism displacement in time.",
          style=dict(color='grey', fontSize='small'))
    ]),
    html.Div([
        html.H2("Prism displacement"),
        dcc.Markdown("""
            Select, using the pictures, the prisms for which you would like to produce displacement plots.
            Selection can be done:
            - by clicking (hold Shift or Ctrl for multiple selection);
            - using the Lasso- or Box-selection tools (on the toolbar).
        """),
        dcc.Graph(id='fig_prism_displacement_selection', 
                  figure=fig_prism_selection) 
    ]),
    html.Div([
    html.Br()
    ]),
    html.Div(id='div_prism_displacement_plots'),
], label='PRISMS')


#--------------------
#    OUTER LEVEL
#--------------------
app.layout = dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Br(),
                        dcc.Markdown('''
                        # Pisa Baptistery
                        ##### *Monitoring Data Analysis and Visualization*
                        '''),
                        html.Br(),
                        dbc.Tabs([
                            tab_info,
                            tab_checks,
                            tab_plan,
                            tab_section,
                            tab_prisms
                        ])
    ])
], lg=dict(width=10, offset=1)),
    dbc.Col(width=1)
])

#======================
#      CALLBACKS
#======================
#----------
# INFO tab
#----------
#---Gantt chart
@app.callback(Output('fig_gantt', 'figure'),
             Input('checklist_gantt', 'value'))
def callFigureGantt(which):
    return figureGantt(which)


#------------
# PLAN tab
#------------
#---Update daterange text in settings pane
@app.callback(Output('text_prism_plan_daterange', 'children'),
             Input('slider_prism_plan_daterange', 'value'))
def callTextPrismPlanDaterange(daterange):
    start = str(prism_data.index[daterange[0]])
    end = str(prism_data.index[daterange[1]])
    text = "From  " + start[:10] + "  to  " + end[:10] 
    return text

#---Update scalefactor text in settings pane
@app.callback(Output('text_prism_plan_scalefactor', 'children'),
             Input('slider_prism_plan_scalefactor_log', 'value'),
             Input('slider_prism_plan_scalefactor_dec', 'value'))
def callTextPrismPlanScalefactor(exp, factor):
    s = scaleFactorCalc(exp, factor)
    return "Displacement scale factor = {}".format(s)

#---Update prism plan plot
@app.callback(Output('fig_prism_plan', 'figure'),
             Input('slider_prism_plan_daterange', 'value'),
             Input('slider_prism_plan_scalefactor_log', 'value'),
             Input('slider_prism_plan_scalefactor_dec', 'value'),
             Input('checklist_prism_plan_floor', 'value'))
def callFigurePrismPlan(daterange, scale_log, scale_dec, floor):
    scalefactor = scaleFactorCalc(scale_log, scale_dec)
    return figurePrismPlan(daterange, scalefactor, floor)


#---------------------
#    SECTION tab
#---------------------
#---Update daterange text in settings pane
@app.callback(Output('text_prism_section_daterange', 'children'),
             Input('slider_prism_section_daterange', 'value'))
def callTextPrismSectionDaterange(daterange):
    start = str(prism_data.index[daterange[0]])
    end = str(prism_data.index[daterange[1]])
    text = "From  " + start[:10] + "  to  " + end[:10] 
    return text

#---Update scalefactor text in settings pane
@app.callback(Output('text_prism_section_scalefactor', 'children'),
             Input('slider_prism_section_scalefactor_log', 'value'),
             Input('slider_prism_section_scalefactor_dec', 'value'))
def callTextPrismSectionScalefactor(exp, factor):
    s = scaleFactorCalc(exp, factor)
    return "Displacement scale factor = {}".format(s)

#--Update section selection plot
@app.callback(Output('fig_prism_section_selection', 'figure'),
             Input('slider_prism_section_selection', 'value'))
def callFigurePrismSectionSelection(selection):
    p = str(selection)
    if len(p) == 1:
        p = '0' + p    
    selected_prisms = selectPrismSection(p)
    return figureSectionSelection(selected_prisms)

#---Update prism section plot
@app.callback(Output('fig_prism_section', 'figure'),
             Input('slider_prism_section_selection', 'value'),
             Input('slider_prism_section_daterange', 'value'),
             Input('slider_prism_section_scalefactor_log', 'value'),
             Input('slider_prism_section_scalefactor_dec', 'value'),
             Input('checklist_prism_section_fixedbase', 'value'))
def callFigurePrismSection(selection, daterange, scale_log, scale_dec, fixedbase):
    s = scaleFactorCalc(scale_log, scale_dec)
    if len(fixedbase) == 1:
        f = True
    else:
        f = False
    p = str(selection)
    if len(p) == 1:
        p = '0' + p
    selected_prisms = selectPrismSection(p)
    return figurePrismSection(selected_prisms, daterange, s, f)

#---Plot relative displacements
@app.callback(Output('div_relative_displacement_plots', 'children'),
             Input('slider_prism_section_selection', 'value'))
def callFigureRelativeDisplacements(selection):
    p = str(selection)
    if len(p) == 1:
        p = '0' + p
    selected_prisms = selectPrismSection(p)
    selected_prisms.sort()
    links = [
        [0,1], [2,3], [4,5], [6,7],
        [0,2], [1,3], [4,6], [5,7]
    ]
    couples = [[selected_prisms[l[0]], selected_prisms[l[1]]] for l in links]
    
    children = []
    for c in couples:
        row = dbc.Row([
            dbc.Col([dcc.Graph(figure=figureSectionRelativeDisplacements(c))], width=9),
            dbc.Col([dcc.Graph(figure=figurePrismCoupleSelection(c))], width=3)         
        ], align='center')
        children.append(row)
        
    return children

#---------------------
#    PRISMS tab
#---------------------
#---Select prisms and plot their displacement
@app.callback(Output('div_prism_displacement_plots', 'children'),
             Input('fig_prism_displacement_selection', 'selectedData'))
def callDivPrismDisplacement(selectedData):
    try:
        prisms = [el['customdata'] for el in selectedData['points']]
        children = [dcc.Markdown('''
            In each plot, you can select which traces to exclude or include by clicking on their legend entries. You can isolate a trace by double-clicking it.
        ''')]
        children += [dcc.Graph(figure=figurePrismDisplacement(p)) for p in prisms]
    except:
        children = dcc.Markdown('Select at least one prism.')
    return children


#=====================
#      RUN APP
#=====================
if __name__ == '__main__':
    app.run_server(debug=True)
