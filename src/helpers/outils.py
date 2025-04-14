
import copy
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read_geometry(filename):
    """
    Function to read and parse geometry file from txt
    """
    nodes, lines, planes, color_planes = dict(), dict(), dict(), dict()
    line_id, plane_id, color_id = 1, 1, 1
    mode = None  # Track which section we are in

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            # Identify sections
            if "NODES" in line:
                mode = "nodes"
                continue
            elif "LINES" in line:
                mode = "lines"
                continue
            elif "SENSORS" in line:
                mode = "sensors"
                continue
            elif "PLANES" in line:
                mode = "planes"
                continue
            elif "COLOR" in line:
                mode = "color"
                continue
            elif not line or line.startswith("//"):
                continue  # Skip empty lines and comments

            # Parse Nodes
            if mode == "nodes":
                parts = line.split()
                node_id = int(parts[0])
                x, y, z = map(float, parts[1:])
                nodes[str(node_id)] = (x, y, z)

            # Parse Lines
            elif mode == "lines":
                parts = list(map(int, line.split()))
                if len(parts) == 2:
                    lines[str(line_id)] = (str(parts[0]), str(parts[1]))
                    line_id += 1

            # Parse Planes
            elif mode == "planes":
                parts = list(map(int, line.split()))
                if len(parts) >= 3:  # At least 3 points
                    planes[str(plane_id)] = [str(p) for p in parts]
                    plane_id += 1

            # Parse Planes
            elif mode == "color":
                parts = list(map(int, line.split()))
                if len(parts) == 4:  # At least 3 points
                    color_planes[str(color_id)] = [str(p) for p in parts]
                    color_id += 1

    return nodes, lines, planes, color_planes


def read_sensors_from_txt_file(filename):
    """
    Function Duties:
        Extracts the sensors from the given file.
    Input:
        filename (str): Path to the text file.
    Output:
        sensors: dictionary containing the sensors
            (related to the file geometry).
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Locate the start of the SENSORS section
    start_marker = "SENSORS"
    start_index = None
    for i, line in enumerate(lines):
        if start_marker in line:
            start_index = i+1  # Skip the section title and the separator line
            break

    if start_index is None:
        raise ValueError("SENSORS section not found in the file.")

    # Read the data lines until the next empty line or section
    sensors = dict()
    for line in lines[start_index:]:
        line = line.strip()
        if not line and len(sensors) > 0:  # Stop reading at an empty line
            break
        line = line.split()
        if len(line) == 1:  # skip the separator indicating the total number of sensors
            continue
        if line:
            ch = len(sensors) + 1
            node = line[0]
            direction = [int(i) for i in line[1:]]
            sensors[f'Channel_{ch}'] = {'node': node, 'dir': direction}
            # if line[0] in list(sensors):  # this node already has a sensor
            #     old_dofs = sensors[line[0]]
            #     new_dofs = [float(i) for i in line[1:]]
            #     sensors[line[0]] = list(np.array(old_dofs) + np.array(new_dofs))
            # sensors[line[0]] = [float(i) for i in line[1:]]

    return sensors


def complex_to_normal_mode(mode, max_dof=50, long=True):
    """Transform a complex mode shape to normal mode shape.
    [From EGM codes]

    The real mode shape should have the maximum correlation with
    the original complex mode shape. The vector that is most correlated
    with the complex mode, is the real part of the complex mode when it is
    rotated so that the norm of its real part is maximized. [1]
    ``max_dof`` and ``long`` arguments are given for modes that have
    a large number of degrees of freedom. See ``_large_normal_mode_approx()``
    for more details.

    Literature:
        [1] Gladwell, H. Ahmadian GML, and F. Ismail.
            "Extracting Real Modes from Complex Measured Modes."
            (avaliable in 'doc' folder)

    :param mode: np.ndarray, a mode shape to be transformed. Can contain a single
        mode shape or a modal matrix `(n_locations, n_modes)`.
    :param max_dof: int, maximum number of degrees of freedom that can be in
        a mode shape. If larger, ``_large_normal_mode_approx()`` function
        is called. Defaults to 50.
    :param long: bool, If True, the start in stepping itartion is altered, the
        angles of rotation are averaged (more in ``_large_normal_mode_approx()``).
        This is needed only when ``max_dof`` is exceeded. The normal modes are
        more closely related to the ones computed with an entire matrix. Defaults to True.
    :return: normal mode shape
    """
    if mode.ndim == 1:
        mode = mode[None, :, None]
    elif mode.ndim == 2:
        mode = mode.T[:, :, None]
    else:
        raise Exception(f'`mode` must have 1 or 2 dimensions ({mode.ndim}).')

    # if mode.shape[1] > max_dof   --> Computationally expensive
    if mode.shape[1] > max_dof:
        return _large_normal_mode_approx(mode[:, :, 0].T, step=int(np.ceil(mode.shape[1] / max_dof)) + 1, long=long)

    # 1. Normalize modes so that norm == 1.0
    _norm = np.linalg.norm(mode, axis=1)[:, None, :]
    mode = mode / _norm

    # 2. Obtain U matrix
    mode_T = np.transpose(mode, [0, 2, 1])
    U = np.matmul(np.real(mode), np.real(mode_T)) + \
        np.matmul(np.imag(mode), np.imag(mode_T))

    # Modification to operate without nan values (otherwise np.linalg.eig raise error)
    nan_mode = np.all(np.isnan(U), axis=(1, 2))
    nan_index = np.where(nan_mode)[0]
    if nan_index.size > 0:
        not_nan = [not (i) for i in nan_mode]
        U_copy = U[not_nan, :, :]
    else:
        U_copy = U

    # 3. Obtain eigenvectors & eigenvalues and choose eigenvector associated to max eigenvalue
    val, vec = np.linalg.eig(U_copy)
    # modification to get as a result mode=0 for nan values [spureous modes]
    if nan_index.size > 0:
        val_aux = np.empty((np.shape(val)[0]+len(nan_index), np.shape(val)[1]))
        vec_aux = np.empty(
            (np.shape(U_copy)[0]+len(nan_index), np.shape(U_copy)[1], np.shape(U_copy)[2]))
        val_aux[not_nan, :] = val
        vec_aux[not_nan, :, :] = vec
        for j in nan_index:
            val_aux[not_nan, :] = np.zeros((np.shape(U_copy)[1]))
            vec_aux[j, :, :] = np.zeros(
                (np.shape(U_copy)[1], np.shape(U_copy)[2]))
        i = np.argmax(np.real(val_aux), axis=1)
        normal_mode = np.real([v[:, _] for v, _ in zip(vec_aux, i)]).T
    else:  # in normal cases we are here
        i = np.argmax(np.real(val), axis=1)
        normal_mode = np.real([v[:, _] for v, _ in zip(vec, i)]).T

    return normal_mode


def solve_linearized_RBM(data, expected_ratio=1e-3):
    """
    Solves a linearized 2D Rigid Body Motion (RBM) problem based on known displacements.
    Assumes small displacements: dx ≈ u - θ·y, dy ≈ v + θ·x.
    If displacements are not small enough, they are scaled down before solving and scaled up afterward.

    Input:
    ----------
    data : dict
        Dictionary of points. Each key is a point label, and each value is a dictionary:
            {
                'x': float,
                'y': float,
                'dx': float or None,
                'dy': float or None
            }
        At least one dx and one dy must be provided, with a total of at least 3 known displacements.
    expected_ratio: float, optional
        Expected ratio of dx/x or dy/y to ensure linearization is possible;
        If the maximum ratio is higher than this value, the displacements are scaled down.

    Returns
    -------
    u : float
        Horizontal translation component.
    v : float
        Vertical translation component.
    theta : float
        Rotation angle (in radians).
    data_solved : dict
        Same structure as input `data`, but with all dx and dy values filled in (rescaled).
    sf: float
        scaling factor; if non linearized final displacements want to be obtained,
        divide, u, v, theta by sf

    Remark: data_solved is the linearized version; if a more accurate output result respecting
    RBM were required, a least squares optimization should be performed to obtain the sf such that
    output displacements match the input ones
    """
    # 1. Scale data to ensure small displacements
    ratio_x, ratio_y = list(), list()
    for point in data:
        x, y, dx, dy = data[point]['x'], data[point]['y'], data[point]['dx'], data[point]['dy']
        if dx is not None and x != 0:
            ratio_x.append(dx/x)
        if dy is not None and y != 0:
            ratio_y.append(dy/y)

    if len(ratio_x) == 0:
        raise ValueError('Error: dx must be specified for at least one point')
    if len(ratio_y) == 0:
        raise ValueError('Error: dy must be specified for at least one point')
    elif len(ratio_x + ratio_y) < 3:
        raise ValueError('Error: at least 3 displacements must be specified')

    max_ratio = np.max(np.abs(ratio_x + ratio_y))

    if max_ratio > expected_ratio:
        sf = max_ratio/expected_ratio
    else:
        sf = 1

    for point in data:
        dx, dy = data[point]['dx'], data[point]['dy']
        if dx is not None:
            data[point]['dx'] = dx / sf
        if dy is not None:
            data[point]['dy'] = dy/sf

    # 2. Linear system of equations
    # Build the linear system of equations:
    A = []
    y_vec = []
    for point in data:
        x, y, dx, dy = data[point]['x'], data[point]['y'], data[point]['dx'], data[point]['dy']
        if dx is not None:
            A.append([1, 0, -y])
            y_vec.append(dx)

        if dy is not None:
            A.append([0, 1, x])
            y_vec.append(dy)

    A = np.array(A)
    y_vec = np.array(y_vec)

    # Solve the linear system
    if np.linalg.matrix_rank(A) < 3:
        print('Warning: The system is underdetermined')
    if A.shape[0] == 3:  # exact solution
        x_solved = np.linalg.solve(A, y_vec)
    else:  # least squares solution
        x_solved = np.linalg.lstsq(A, y_vec, rcond=None)[0]

    A_full = np.array([[1, 0, -data[point]['y']] for point in data] +
                      [[0, 1, data[point]['x']] for point in data])
    y_full = A_full @ x_solved

    # Build final data_solved dictionary (important: rescale values)
    data_solved = copy.deepcopy(data)
    for idx, point in enumerate(data):
        x, y, dx, dy = data[point]['x'], data[point]['y'], data[point]['dx'], data[point]['dy']
        dx_solved, dy_solved = y_full[idx], y_full[len(data) + idx]
        data_solved[point]['dx'] = dx_solved*sf
        data_solved[point]['dy'] = dy_solved*sf

    u, v, theta = x_solved*sf

    return u, v, theta, data_solved, sf


def apply_translation_rotation_v2(P, translation_vector, theta):
    """
    Function Duties:
    Obtains the final position of a point P after applying a translation and rotation
    from a centre of rotation point.
    Parameters:
        - P: np.array, initial point position w.r.t. centre of rotation
        - translation_vector: np.array, translation vector
        - theta: rotation angle
    Returns:
        - P_f: np.array, final point position w.r.t. initial centre of rotation
    """
    rotation_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    P_f = translation_vector + rotation_matrix @ P

    return P_f


def plot_structure_modeshape(nodes, lines, planes, sensors_acc=None, sf_arrow=5,
                             numbers_arrow=False,
                             displaced_nodes=None,
                             plot_undeformed=True, plot_planes_undeformed=True):
    """
    Plots a 3D schematic structure and its displacement mode shapes.

    Parameters:
    -----------
    nodes : dict
        Dictionary containing node coordinates {node_id: (x, y, z)}.
    lines : dict
        Dictionary containing line connections {line_id: (start_node, end_node)}.
    planes : dict
        Dictionary containing plane definitions {plane_id: [node1, node2, node3, node4]}.
    sensors_acc : dict, optional
        Dictionary containing sensor definitions for acceleration measurements.  
        Keys are channel identifiers (e.g., 'Channel_1'), and values are dictionaries  
        with the associated node and the measurement direction vector.  
        Format: {channel_id: {'node': node_id, 'dir': [dx, dy, dz]}}.
    sf_arrow: int, optional [applies if sensors_acc is not None]
        Scaling factor for the arrow length of the sensors. Default is 5.
    numbers_arrow: bool, optional [applies if sensors_acc is not None]
        If True, the sensor channel numbers are displayed next to the arrows.
    displaced_nodes : dict, optional
        Dictionary with deformed node coordinates {node_id: (x, y, z)}.
        If None, the undeformed configuration is used.
    plot_undeformed : bool, optional (default=True)
        If True, plots the undeformed frame in light gray for reference.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axis object containing the plot.
    """
    fig = plt.figure()
    fig.patch.set_facecolor('white')  # Ensure white background
    ax = fig.add_subplot(111, projection='3d')

    if displaced_nodes is None:
        plot_undeformed = False
        displaced_nodes = nodes

    if not sensors_acc:
        sf_arrow = None

    # Plot Undeformed Frame (Low Opacity)
    if plot_undeformed:
        # Plot Planes (Slabs)
        if plot_planes_undeformed:
            for _, plane in planes.items():
                plane_vertices = [nodes[pid]
                                  for pid in plane]  # Get coordinates
                ax.add_collection3d(Poly3DCollection(
                    [plane_vertices], color='gray', alpha=0.1))

        for _, (start, end) in lines.items():
            x_vals = [nodes[start][0], nodes[end][0]]
            y_vals = [nodes[start][1], nodes[end][1]]
            z_vals = [nodes[start][2], nodes[end][2]]
            ax.plot(x_vals, y_vals, z_vals, color='gray',
                    linewidth=1, alpha=0.3)  # Low opacity

    # Plot Deformed
    # First we plot planes (as they affect the visibility of the lines)
    for _, plane in planes.items():
        plane_vertices = [displaced_nodes[pid] for pid in plane]
        ax.add_collection3d(Poly3DCollection(
            [plane_vertices], color='blue', alpha=0.2))

    # Plot Deformed Frame (Fully Visible):
    color = 'b'
    linewidth = 1
    alpha = 1
    for line, (start, end) in lines.items():
        x_vals = [displaced_nodes[start][0], displaced_nodes[end][0]]
        y_vals = [displaced_nodes[start][1], displaced_nodes[end][1]]
        z_vals = [displaced_nodes[start][2], displaced_nodes[end][2]]
        ax.plot(x_vals, y_vals, z_vals, color=color,
                linewidth=linewidth, alpha=alpha)

    # Plot arrow
    if sensors_acc:
        for channel, sensor_info in zip(list(sensors_acc), sensors_acc.values()):
            node = sensor_info['node']
            direction = sensor_info['dir']

            x_start, y_start, z_start = displaced_nodes[node]

            dx, dy, dz = direction  # magnitud unitaria en esa dirección
            dx = dx * sf_arrow
            dy = dy * sf_arrow
            dz = dz * sf_arrow

            # Plot arrow
            ax.quiver(x_start, y_start, z_start, dx, dy, dz,
                    color='red', length=max(dx, dy, dz), normalize=True)
            if numbers_arrow:
                ax.text(x_start+dx, y_start+dy, z_start+dz, channel)

    # Remove background grid and panes, set axis limits, etc.
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Set pane edges to fully transparent
    ax.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    # Set pane faces to fully transparent
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Make x-axis invisible
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Make y-axis invisible
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Make z-axis invisible
    # ax.set_frame_on(False)

    # set x limits (ax.axis('equal') avoids zooming so we discard it)
    x_limits = [min(node[0] for node in displaced_nodes.values()),
                max(node[0] for node in displaced_nodes.values())]
    y_limits = [min(node[1] for node in displaced_nodes.values()),
                max(node[1] for node in displaced_nodes.values())]
    z_limits = [min(node[2] for node in displaced_nodes.values()),
                max(node[2] for node in displaced_nodes.values())]
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    max_range = max(x_limits[1] - x_limits[0], y_limits[1] -
                    y_limits[0], z_limits[1] - z_limits[0]) / 2.0
    ax.set_xlim([x_mid - max_range, x_mid + max_range])
    ax.set_ylim([y_mid - max_range, y_mid + max_range])
    ax.set_zlim([z_mid - max_range, z_mid + max_range])

    # Set view angle
    plt.draw()

    return fig, ax
