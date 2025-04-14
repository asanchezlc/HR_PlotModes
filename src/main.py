
import copy
import json
import numpy as np
import os

import helpers.outils as outils
import matplotlib.pyplot as plt

from pathlib import Path


"""
File used to quickly plot the HR mode shapes
"""

# 1. DATA DEFINITION
# A) Paths and files
current_dir = Path(__file__).resolve().parent
figpath = os.path.join(current_dir, "figures")
datapath = os.path.join(current_dir, "data")
filename = "geometry_HR.txt"
modaldata_filename = "oma_data_api.json"

# B) Plotting parameters
# scale factors per mode shape
sf_modeshapes = dict()
sf_modeshapes['Mode_1'] = 5
sf_modeshapes['Mode_2'] = 5
sf_modeshapes['Mode_3'] = 5
sf_modeshapes['Mode_4'] = 5

# viewing angles for each mode shape
# views = {'Mode_1': (11.3, -84.4), 'Mode_2': (11.3, -84.4), 'Mode_3': (11.3, -84.4),
#          'Mode_4': (11.3, -84.4), 'Mode_5': (11.3, -84.4)}

# 2. LOAD DATA
# A) Geometry data
# Read the geometry related to accelerometers from the file
nodes, lines, planes, color_planes = outils.read_geometry(os.path.join(datapath, filename))
sensors_acc = outils.read_sensors_from_txt_file(os.path.join(datapath, filename))

# Plot geometry
fig, ax = outils.plot_structure_modeshape(nodes, lines, color_planes, sensors_acc=sensors_acc,
                                          numbers_arrow=True, sf_arrow=5)
figname = 'Geometry.pdf'
fig.savefig(os.path.join(figpath, figname), dpi=300)

# B) OMA data
# Retrieve OMA data
with open(os.path.join(datapath, modaldata_filename), 'r') as json_file:
    modaldata = json.load(json_file)
oma = modaldata[0]
frequencies_oma = np.array([oma['modos'][i]['frequency'] for i in range(len(oma['modos']))])
n_modes = len(oma['modos'])
n_sensors = len(oma['sensors'])
Phi_oma = np.zeros((n_sensors, n_modes), dtype=complex)
for i in range(n_modes):
    phi_list = [oma['modos'][i]['phi'][j]['real'] + 1j*oma['modos'][i]['phi'][j]['imag'] for j in range(n_sensors)]
    Phi_oma[:, i] = np.array(phi_list)
Phi_oma = outils.complex_to_normal_mode(Phi_oma)

# 3. PLOT EACH MODE SHAPE
for n_mode in range(np.shape(Phi_oma)[1]):
    print(f'Mode {n_mode+1}')
    print('--------------------')
    # 3.1. GET DISPLACEMENTS ACCORDING TO RBM
    # Retrieve displacement mode shapes in each node
    displacement_nodes = dict()
    for i, ch in enumerate(sensors_acc):
        node = sensors_acc[ch]['node']
        value = np.array(sensors_acc[ch]['dir']) * Phi_oma[i, n_mode]
        if node in displacement_nodes:
            displacement_nodes[node] = (displacement_nodes[node][0] + value[0],
                                        displacement_nodes[node][1] + value[1], displacement_nodes[node][2] + value[2])
        else:
            displacement_nodes[node] = (value[0], value[1], value[2])
    for node in nodes:
        if node not in displacement_nodes:
            displacement_nodes[node] = (0, 0, 0)

    sf_modeshapes_i = sf_modeshapes[f'Mode_{n_mode + 1}']
    for plane, nodes_group in planes.items():
        # get the input data for RBM problem
        centroid = np.mean([nodes[node] for node in nodes_group], axis=0)
        centre = np.array([centroid[0], centroid[1]])
        data = dict()
        for i, node in enumerate(nodes_group):
            x, y = nodes[node][0], nodes[node][1]
            dx, dy = displacement_nodes[node][0], displacement_nodes[node][1]
            if dx == 0:
                dx = None
            if dy == 0:
                dy = None
            data[node] = {'x': x - centre[0], 'y': y - centre[1], 'dx': dx, 'dy': dy}

        # solve the linearized RBM
        u, v, theta, data_solved, sf = outils.solve_linearized_RBM(data)

        # Final positions given a scaling plot factor and non-linear rotation
        sf_modeshapes_i = sf_modeshapes[f'Mode_{n_mode + 1}'] * sf
        data_scaled = copy.deepcopy(data)
        u, v, theta = u/sf, v/sf, theta/sf
        translation_vector, theta = sf_modeshapes_i*np.array([u, v]), sf_modeshapes_i*theta
        for idx, point in enumerate(data):
            P = np.array([data[point]['x'], data[point]['y']])
            P_f = outils.apply_translation_rotation_v2(P, translation_vector, theta)
            dx, dy = P_f - P
            data_scaled[point]['dx'] = dx
            data_scaled[point]['dy'] = dy

        # Get the displacement nodes
        for point in data_scaled:
            displacement_nodes[point] = (data_scaled[point]['dx'], data_scaled[point]['dy'], 0)

    # Compute displaced node positions and plot them
    displaced_nodes = dict()
    for node in nodes:
        if node in displacement_nodes:
            dx, dy, dz = displacement_nodes[node]
        else:
            dx, dy, dz = 0, 0, 0
        displaced_nodes[node] = (nodes[node][0] + dx, nodes[node][1] + dy, nodes[node][2] + dz)

    fig, ax = outils.plot_structure_modeshape(nodes, lines, color_planes, displaced_nodes=displaced_nodes, plot_undeformed=True)
    # ax.view_init(elev=views[f'Mode_{n_mode + 1}'][0], azim=views[f'Mode_{n_mode + 1}'][1])
    plt.draw()
    figname = f'DispMode_{n_mode + 1}.pdf'
    fig.savefig(os.path.join(figpath, figname), dpi=300)
