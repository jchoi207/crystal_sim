from pymatgen.analysis.diffraction.tem import TEMCalculator
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math as m
from decimal import Decimal, ROUND_HALF_UP

import time
import warnings
from joblib import dump, load


def get_crystals_from_file(filename: str, api_key: str) -> tuple:
    """
    Retrieves crystal structure properties, crystal system classifications, and material IDs from a file using the Materials Project API.

    Args:
        filename (str): Path to the .txt file containing Material Project (MPR) material IDs, one per line.
        api_key (str): API key for accessing the Materials Project database.

    Returns:
        tuple:
            - crystal_list (list): A list of crystal structures.
            - system_list (list): A list of crystal system classifications, which can be one of the following:
                'Triclinic', 'Trigonal', 'Orthorhombic', 'Cubic', 'Monoclinic', 'Tetragonal', 'Hexagonal'.
            - material_id_list (list): A list of tuples, where each tuple contains a material ID and its formula, e.g., [(id_1, formula_1), ... , (id_n, formula_n)].
    """
    with open(filename, 'r') as file:
        materials_random_order = [material_id.strip() for material_id in file]

    with MPRester(api_key=api_key) as mpr:
        crystals = mpr.materials.search(
            material_ids=materials_random_order,
            fields=['structure', 'symmetry', 'material_id', 'formula_pretty']
        )

    crystal_list = [crystal.structure for crystal in crystals]
    system_list = [str(crystal.symmetry.crystal_system)
                   for crystal in crystals]
    material_id_list = [(crystal.material_id, crystal.formula_pretty)
                        for crystal in crystals]

    return crystal_list, system_list, material_id_list


def get_cartesian_beam_directions(upper: int = 1) -> list:
    """
    Generates a list of beam directions in Cartesian coordinates based on spherical coordinates.

    Args:
        upper (int): The number of divisions of the 1/8 sphere. The resulting number of beam directions will be (upper + 1)**2.

    Returns:
        list: A list of unique beam directions as tuples (x, y, z) rounded to three decimal places.
    """

    beam_direction_list = []
    for phi_index in range(upper + 1):
        for theta_index in range(upper + 1):
            p = m.pi / 2 * phi_index / upper
            t = m.pi / 2 * theta_index / upper
            x = round(m.sin(t) * m.cos(p), 3)
            y = round(m.sin(t) * m.sin(p), 3)
            z = round(m.cos(t), 3)
            beam_direction_list.append([x, y, z])

    return np.unique(beam_direction_list, axis=0).tolist()


def my_atan(x_y: list) -> float:
    """
    Computes the arctangent of y/x, considering the sign and quadrant of the point.

    Args:
        x_y (list): A list containing x and y Cartesian coordinates.

    Returns:
        float: The arctangent of y/x, rounded to three decimal places, in the range [0, 2π).
    """

    if x_y[0] == 0:
        return round(m.pi / 2 if x_y[1] > 0 else m.pi * 1.5, 3)
    if x_y[1] == 0:
        return round(float(0) if x_y[0] > 0 else m.pi, 3)

    atan_value = m.atan(x_y[1] / x_y[0])
    if x_y[0] * x_y[1] > 0:  # Quadrants 1 or 3
        return round(atan_value + (m.pi if x_y[0] < 0 else 0), 3)
    else:  # Quadrants 2 or 4
        return round(atan_value + (m.pi if x_y[1] > 0 else 2 * m.pi), 3)


def get_rounded_position(data: list, position: bool) -> list:
    """
    Rounds the position or intensity data to three decimal places.

    Args:
        data (list/float): The data to be rounded.
        position (bool): If True, rounds a list of values; otherwise, rounds a single value.

    Returns:
        list/float: The rounded data.
    """

    def round_decimal(value):
        return float(Decimal(value).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))

    if position:
        return [round_decimal(x) for x in data]
    else:
        return round_decimal(data)


def plot_patterns(dfs: list, beam_directions: list, system_list: list, material_id: str) -> None:
    """
    Plots diffraction patterns and vectors from the center of the beam to the five closest spots for each pattern.

    Args:
        dfs (list): A list of DataFrames containing diffraction pattern data, each with 'Position' and 'Intensity (norm)' columns.
        beam_directions (list): A list of beam direction vectors.
        system_list (list): A list of crystal system classifications corresponding to each DataFrame.
        material_id (str): The material ID to be used as the title of the plot.
    """
    count = 0
    l = 3
    fig, ax = plt.subplots(nrows=l, ncols=l, figsize=(5 * l, 5 * l))
    fig.suptitle(f"{material_id} \n {system_list} \n Electron Diffraction Pattern")

    for i in range(l):
        for j in range(l):
            if count == len(beam_directions):
                break
            df = dfs[count]
            x, y = zip(*df['Position'])
            x = [float(i) for i in x]
            y = [float(i) for i in y]

            ax[i][j].grid(linewidth=0.35, zorder=1)
            ax[i][j].scatter(0, 0, c='red', s=30, zorder=2)
            ax[i][j].scatter(x, y, c="w", s=df['Intensity (norm)']
                             * 25, alpha=df['Intensity (norm)'], zorder=2)
            ax[i][j].set_title(f"{beam_directions[count]}")
            ax[i][j].set_facecolor("#000000")
            ax[i][j].set_ylabel('e-10 m')
            ax[i][j].set_xlabel('e-10 m')
            ax[i][j].set_xlim([-5, 5])
            ax[i][j].set_ylim([-5, 5])

            df_5 = df.sort_values('Angle', ascending=True).iloc[0:5]
            x_cor_list, y_cor_list = zip(*df_5['Position'])
            x_cor_list, y_cor_list = np.array(x_cor_list), np.array(y_cor_list)

            for k in range(len(df_5['Position'])):
                ax[i][j].annotate(f"({round(df_5['Magnitude'].iloc[k], 1)}, \
                                  {round(df_5['Angle'].iloc[k], 1)})", (x_cor_list[k], y_cor_list[k]), color='w', fontsize=8.5)

            ax[i][j].scatter(x_cor_list, y_cor_list, c="w", s=df_5['Intensity (norm)']
                             * 25, alpha=df_5['Intensity (norm)'], zorder=3)
            ax[i][j].quiver(np.zeros(5), np.zeros(
                5), x_cor_list, y_cor_list, scale=1, scale_units='xy', color='g', zorder=1.5)
            count += 1


def get_dp_from_beam_direction(beam_direction: tuple[float, float, float], my_crystal_structure: Structure) -> pd.DataFrame:
    """
    Retrieves and processes the diffraction pattern data for a given crystal structure and beam direction.

    Args:
        beam_direction (tuple[float, float, float]): The beam direction vector (x, y, z).
        my_crystal_structure (Structure): The crystal structure information.

    Returns:
        pd.DataFrame: A DataFrame containing the top 10 diffraction spots sorted by 
                     intensity. Includes columns for position (x, y), intensity, magnitude, and angle.
    """

    my_tem = TEMCalculator(beam_direction=beam_direction, voltage=80)
    df = my_tem.get_pattern(SpacegroupAnalyzer(
        my_crystal_structure).get_conventional_standard_structure())
    warnings.filterwarnings(
        "ignore", message="invalid value encountered in arccos")
    df = df.drop(columns=['Interplanar Spacing', '(hkl)', 'Film radius'])
    df = df.sort_values('Intensity (norm)', ascending=False).iloc[0:10][:]
    df['Position'] = df['Position'].apply(
        lambda pos: get_rounded_position(pos, True))
    df['Intensity (norm)'] = df['Intensity (norm)'].apply(
        lambda intens: get_rounded_position(intens, False))
    df.reset_index(drop=True, inplace=True)
    return df


def get_polar_coordinates(df: pd.DataFrame) -> tuple:
    """
    Computes polar coordinates and angles for the top 5 closest diffraction spots.

    Args:
        df (pd.DataFrame): DataFrame containing diffraction spots with 'Position' column.

    Returns:
        tuple:
            - magnitudes (list): A list of magnitudes of the top 5 closest spots.
            - angles (list): A list of angular differences between consecutive top 5 closest spots.
    """
    df['Magnitude'] = df['Position'].apply(
        lambda x: round((x[0]**2 + x[1]**2)**0.5, 3))
    df['Angle'] = df['Position'].apply(lambda x: my_atan(x))
    df_top_5 = df.sort_values('Angle', ascending=True).iloc[0:5]

    magnitudes = list(df_top_5['Magnitude'])
    angles = list(np.round(np.diff(df_top_5['Angle']), decimals=3))

    return magnitudes, angles


def write_data_to_txt(filename: str, data: list) -> None:
    """
    Writes a list of vectors to a .txt file, with each vector formatted as 'vector((0,0,0),[x,y,z])'.

    Args:
        filename (str): The base name of the file to write to (without extension).
        data (list): A list of vectors to be written to the file.
    """
    with open(f"{filename}.txt", "x") as f:
        for d in data:
            f.write(f"vector((0,0,0),{str(d)})\n")


def get_preprocessed_data(crystal_list: list, system_list: list, material_id_list: list,
                          beam_directions: list, plot: bool = False, vectors: bool = True) -> list:
    """
    Preprocesses crystal data, generates features and labels, and optionally plots diffraction patterns.

    Args:
        crystal_list (list): A list of crystal structures.
        system_list (list): A list of crystal system classifications.
        material_id_list (list): A list of tuples containing material IDs and formulas.
        beam_directions (list): A list of beam direction vectors.
        plot (bool, optional): Whether to plot diffraction patterns. Defaults to False.
        vectors (bool, optional): Whether to use vectors for features. Defaults to True.

    Returns:
        tuple:
            - features (list): A list of feature vectors for each crystal.
            - labels_regression (list): A list of unit cell parameters for regression.
            - labels_classification (list): A list of crystal system classifications for classification.
            - material_id_list (list): A list of tuples containing material IDs and formulas.
    """
    features = []
    labels_regression = []
    labels_classification = system_list
    times = []
    base_time = time.time()
    start_time = time.time()
    length = len(crystal_list)

    for i in range(length):
        my_crystal_structure = crystal_list[i]
        struct = my_crystal_structure.as_dict()['lattice']

        if i % 25 == 0:
            end_time = time.time()
            print(f"_________________{i*100//length}% complete_________________")
            time_diff = end_time - start_time
            print(f"Time elapsed: {round(end_time - base_time, 3)} s")
            print(f"Time diff: {round(time_diff, 3)} s")
            times.append(time_diff)
            start_time = time.time()

            print(f"{i}: {material_id_list[i]} ")

        parameters = [round(struct[unit_parameter], 2) for unit_parameter in [
            'a', 'b', 'c', 'alpha', 'beta', 'gamma']]
        labels_regression.append(parameters)

        dfs = [get_dp_from_beam_direction(
            direction, my_crystal_structure) for direction in beam_directions]

        features_i = []
        if vectors:
            for df in dfs:
                magnitudes, angles = get_polar_coordinates(df)
                features_i += magnitudes + angles
        else:
            for df in dfs:
                flattened = [coord for pair in df['Position']
                             for coord in pair]
                features_i += flattened

        features.append(features_i)

        if vectors and plot:
            plot_patterns(dfs, beam_directions,
                          labels_classification[i], material_id_list[i])

    print("___________________________________")
    print(f"Time elapsed: {round(time.time() - base_time, 3)} s")
    print("100% complete")
    print("______________________________________________________________________")

    return features, labels_regression, labels_classification, material_id_list