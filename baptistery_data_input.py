# IMPORTS
#------------
import numpy as np
import pandas as pd
from geopy.distance import distance as geodist


# DATA IMPORT
#--------------
#---PRISMS
def readPrismsData():
    # Read info about prisms for column indices
    prisms = np.loadtxt('Data/prismi.csv', delimiter='\t', dtype='object', max_rows=1)
    prisms = [el for el in prisms if el!='']
    directions = ['x', 'y', 'z']*len(prisms)
    tmp = []
    for p in prisms:
        tmp += [p]*3
    prism_multiindex = np.array([tmp, directions]).transpose()

    # Read prism data
    tmp = pd.read_csv("Data/prismi.csv", delimiter = "\t", index_col= 0, skiprows=1, parse_dates=True)

    # Format prism DataFrame with MultiIndex
    prism_data = pd.DataFrame(tmp.values, index=tmp.index, columns=prism_multiindex)
    prism_data.columns = pd.MultiIndex.from_tuples(prism_data.columns)
    
    return prism_data

#---LEVELLING
def readLevellingData():
    levelling_data = pd.read_csv('Data/levelling.csv', delimiter='\t', index_col='date', parse_dates=True)
    return levelling_data

#---EXTENSIMETERS
def readExtensimetersData():
    # Read extensimeter data (position and temperature) in a temporary DataFrame
    # because there are two separate columns for date and time
    tmp = pd.read_csv('Data/extensimeters.csv', delimiter=';', header=[0,1])

    # Create a new DataFrame with a single index DateTime column
    cols_to_join = tmp.columns[:2]
    dt_index = [tmp[cols_to_join[0]].iloc[i] + ' ' + tmp[cols_to_join[1]].iloc[i] for i in range(len(tmp.index))]
    extensimeter_data = pd.DataFrame(tmp[tmp.columns[2:]].values, index=dt_index, columns=tmp.columns[2:])
    extensimeter_data.index = pd.to_datetime(extensimeter_data.index)
    # Removes measurements if temperature == 0°C (because we think they are outliers)
    extensimeter_data = extensimeter_data.replace(0.0, np.nan).dropna().copy()
    
    return extensimeter_data

def readAllData():
    p = readPrismsData()
    l = readLevellingData()
    e = readExtensimetersData()
    
    return p, l, e


# SENSOR LOCATION IMPORT
#---------------------------
def readSensorPositions():
    prism_pos = pd.read_csv('Data/positions/prism_angles.csv', index_col=0, names=['angle', 'radius', 'z'])
    prism_pos['type'] = ['prism']*len(prism_pos)
    levelling_pos = pd.read_csv('Data/positions/levelling_angles.csv', index_col=0, names=['angle','radius', 'z'])
    levelling_pos['type'] = ['level']*len(levelling_pos)
    extensimeter_pos = pd.read_csv('Data/positions/extensimeter_angles.csv', index_col=0, names=['angle','radius', 'z'])
    extensimeter_pos['type'] = ['crack']*len(extensimeter_pos)

    # Single DataFrame with all instrument positions
    positions = pd.concat([prism_pos, levelling_pos, extensimeter_pos], axis=0)
    
    return prism_pos, levelling_pos, extensimeter_pos, positions


# EARTQUAKE DATA
#----------------
def readEarthquakeData():
    earthquake_data = pd.read_csv('Data/earthquakes.csv', index_col=0, parse_dates=True, delimiter=',')
    bapt_lat, bapt_lon = 43.7232378, 10.3944417 
    earthquake_data['Distance'] = [geodist((bapt_lat, bapt_lon), (lat, lon)).km for lat, lon in zip(earthquake_data['Latitude'], earthquake_data['Longitude'])]

    return earthquake_data
