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
import os
from joblib import dump, load


def get_crystals_from_file(filename: str, api_key: str) -> tuple:
    """
    Retrieves crystal structure properties, crystal system classifications, and mp-ids from a file using the Materials Project API.

    Args:
        filename (str): Path to the .txt file containing Material Project (MPR) mp-ids, separated by line breaks.
        api_key (str): API key.

    Returns:
        tuple:
            crystal_list (list): A list of crystal structures.
            space_group_list (list): A list of spacegroup numbers.
            bravais_list (list): A list of Bravais Lattice types, which can be one of the following:
                    'Simple (P)', 'Body centered (I)', 'Face centered (F)'
            system_list (list): A list of crystal system classifications, which can be one of the following:
                    'Triclinic', 'Trigonal', 'Orthorhombic', 'Cubic', 'Monoclinic', 'Tetragonal', 'Hexagonal'.
            material_id_list (list): A list of tuples, where each tuple contains a material ID and its formula, e.g., [(id_1, formula_1), ... , (id_n, formula_n)].
    """
    with open(filename, 'r') as file:
        materials_random_order = [material_id.strip() for material_id in file]

    with MPRester(api_key=api_key) as mpr:
        crystals = mpr.materials.search(
            material_ids=materials_random_order,
            fields=['structure', 'symmetry',
                    'material_id', 'formula_pretty', 'chemsys']
        )

    crystal_list = [crystal.structure for crystal in crystals]
    space_group_list = [crystal.symmetry.number for crystal in crystals]
    bravais_list = [str(crystal.symmetry.symbol)[0] for crystal in crystals]
    system_list = [str(crystal.symmetry.crystal_system)
                   for crystal in crystals]
    material_id_list = [(crystal.material_id, crystal.formula_pretty)
                        for crystal in crystals]

    return (
        crystal_list,
        space_group_list,
        bravais_list,
        system_list,
        material_id_list
    )


def retrieve_crystals_from_api(api_key: str, crystal_system: str, base_dir: str, write: bool = False, **kwargs) -> tuple:
    """
    Retrieves ALL mp-ids that correspond to the crystal system.

    Args:
        api_key (str): API key.
        crystal_system (str): Crystal system of interest.
        base_dir (str): Base directory for the project.
        write (bool): If True, all the ids are written to mp-ids/crystal_system.
        **kwargs: Additional keyword arguments.
            min_size (int): Limit the number of samples to min_size * len(unique(bravais lattices)).

    Returns:
        tuple:
            bravais_dict: Dictionary with keys as Bravais lattice types and values as corresponding mp-ids.
            least_datapoints: Least number of datapoints of all the values in the dictionary.
            api_key: API key.
    """

    bravais_dict = {
        'P': [],  # primitive
        'I': [],  # body centered
        'A': [],  # face centered
        'F': [],  # centered on A
        'C': [],  # centered on C
        'R': [],  # rhombohedral
    }

    conversion_dict = {
        'P': ['Primitive    '],
        'I': ['Body Centered'],
        'A': ['Face Centered'],
        'F': ['Centered on A'],
        'C': ['Centered on C'],
        'R': ['Rhombohedral ']
    }

    with MPRester(api_key=api_key) as mpr:
        crystals = mpr.materials.search(
            elements=["Si", "O"],
            crystal_system=[crystal_system.capitalize()],
            fields=['material_id', 'symmetry']
        )

    for crystal in crystals:
        bravais_dict[str(crystal.symmetry.symbol)[0]
                     ].append(crystal.material_id)

    non_empty_keys = [key for key in list(
        bravais_dict.keys()) if len(bravais_dict[key]) > 0]

    sizes = []

    if write:
        new_folder = os.path.join(base_dir, "mp_ids")

        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        file_dir = os.path.join(new_folder, crystal_system+".txt")
        file = open(file_dir, 'w')

        for non_empty_key in non_empty_keys:
            length_space_group = len(bravais_dict[non_empty_key])
            sizes.append(length_space_group)
            file.write(f"~{non_empty_key}-{length_space_group}\n")

            for value in list(bravais_dict[non_empty_key]):
                file.write(f"{value}\n")
        file.close()

    bravais_dict = {k: v for k, v in bravais_dict.items()
                    if k in non_empty_keys}

    print(f"_________________Summary:_________________\n")

    for x in bravais_dict.keys():
        sizes.append(len(bravais_dict[x]))

    for x, y in zip(non_empty_keys, sizes):
        print(f">> {x} {conversion_dict[x][0]}: number of crystals - {y}")

    least_datapoints = min(sizes)
    print(f">> Total number of crystals: {sum(sizes)}\n")
    print(f">> Smallest number of crystals: {least_datapoints}")

    if kwargs['min_size'] != None:
        min_size = kwargs['min_size']
        print(f">> Overwriting the smallest number of available sample points from {least_datapoints} to {min_size}")
        print(f">> {min_size * len(non_empty_keys)} crystals will be processed\n")
        return (
            bravais_dict,
            min_size,
            api_key
        )
    else:
        min_size = least_datapoints + 1
        while min_size > least_datapoints:
            try:

                min_size = int(input(f"<< How many crystals per Bravais Lattice type would you like to process? \nThis number must necessarily be <= {least_datapoints} to ensure a balanced dataset"))
                if min_size > least_datapoints:
                    print(f">> Invalid input. Enter a number <= {least_datapoints}")
            except ValueError:
                print(">> Invalid input. Please enter a valid integer.")

        print(f">> {min_size * len(non_empty_keys)} crystals will be processed\n")

        return (
            bravais_dict,
            min_size,
            api_key
        )


def get_crystal_info(retrieve_crystals_tuple) -> tuple:
    """
    Retrieves crystal structure properties, crystal system classifications, 
    and material IDs from a dictionary of mp-ids

    Args:
        tuple:
            bravais_dict (dict): See above documentation.
            min_size (int): See above documentation.
            api_key (str): See above documentation.

    Returns:
        crystal_list (list): A list of crystal structures.
        space_group_list (list): A list of spacegroup numbers.
        bravais_list (list): A list of Bravais Lattice types, which can be one of the following:
                'Simple (P)', 'Body centered (I)', 'Face centered (F)'
        system_list (list): A list of crystal system classifications, which can be one of the following:
                'Triclinic', 'Trigonal', 'Orthorhombic', 'Cubic', 'Monoclinic', 'Tetragonal', 'Hexagonal'.
        material_id_list (list): A list of tuples, where each tuple contains a material ID and its formula, e.g., [(id_1, formula_1), ... , (id_n, formula_n)].
    """
    bravais_dict, min_size, api_key = retrieve_crystals_tuple
    crystal_list, space_group_list, bravais_list, system_list, material_id_list = [], [], [], [], []

    for k in bravais_dict.keys():
        shuffled_array = np.array(bravais_dict[k])
        np.random.shuffle(shuffled_array)
        shuffle_list = shuffled_array.tolist()

        with MPRester(api_key=api_key) as mpr:
            crystals = mpr.materials.search(
                material_ids=shuffle_list[:min_size],
                fields=['structure', 'symmetry',
                        'material_id', 'formula_pretty', 'chemsys']
            )

        crystal_list += [crystal.structure for crystal in crystals]
        space_group_list += [crystal.symmetry.number for crystal in crystals]
        bravais_list += [str(crystal.symmetry.symbol)[0]
                         for crystal in crystals]
        system_list += [str(crystal.symmetry.crystal_system)
                        for crystal in crystals]
        material_id_list += [(crystal.material_id, crystal.formula_pretty)
                             for crystal in crystals]

    return (
        crystal_list,
        space_group_list,
        bravais_list,
        system_list,
        material_id_list
    )


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
    fig.suptitle(f"{material_id} \
                 {system_list} \n Electron Diffraction Pattern")

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

    # voltage for HT8700 CFE: 80 kV
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


def get_preprocessed_data(get_crystal_returns: tuple, beam_directions: list, plot: bool = False, vectors: bool = True) -> list:
    """
    Preprocesses crystal data, generates features and labels, and optionally plots diffraction patterns.

    Args:
        tuple:
            crystal_list (list): A list of crystal structures.
            space_group_list (list): A list of space group numbers.
            bravais_list (list): A list of bravais lattice types.
            system_list (list): A list of crystal system classifications.
            material_id_list (list): A list of tuples containing material IDs and formulas.
        beam_directions (list): A list of beam direction vectors.
        plot (bool, optional): Whether to plot diffraction patterns. Defaults to False.
        vectors (bool, optional): Whether to use vectors for features. Defaults to True.

    Returns:
        tuple:
            features (list): A list of feature vectors for each crystal.
            labels_regression (list): A list of unit cell parameters for regression.
            labels_classification_space (list): A list of crystal system space groups.
            labels_classification_bravais (list): A list of bravais lattice types.
            labels_classification_system (list): A list of crystal system classifications.
            material_id_list (list): A list of tuples containing material IDs and formulas.
    """

    crystal_list, space_group_list, bravais_list, system_list, material_id_list = get_crystal_returns
    features = []
    labels_regression = []
    labels_classification_space = space_group_list
    labels_classification_bravais = bravais_list
    labels_classification_system = system_list
    times = []
    base_time = time.time()
    start_time = time.time()
    length = len(crystal_list)

    if length < 10:
        interval_denom = length
    else:
        interval_denom = length // 10

    for i in range(length):
        my_crystal_structure = crystal_list[i]
        struct = my_crystal_structure.as_dict()['lattice']

        if i % (interval_denom) == 0:
            end_time = time.time()
            print(f"_________________{i*100//length}% complete_________________")
            time_diff = end_time - start_time
            print(f">> Crystal No.{i}: {material_id_list[i]} ")
            print(f">> Time elapsed: {round(end_time - base_time, 3)} s")
            print(f">> Time diff: {round(time_diff, 3)} s")
            times.append(time_diff)
            start_time = time.time()

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
                          labels_classification_system[i], material_id_list[i])

    print("___________________________________")
    print(f">> Total time elapsed: {round(time.time() - base_time, 3)} s")
    print(">> 100% complete")
    print("______________________________________________________________________\n")

    return (
        features,
        labels_regression,
        labels_classification_space,
        labels_classification_bravais,
        labels_classification_system,
        material_id_list
    )


def save_data(get_preprocessed_returns, base_dir, new_dir) -> str:
    """
    Saves the returns from get_preprocessed_data , labels and material ids from previous code modules

    Args:
        tuple:
            features (list): list of features.
            labels_regression (list): unit cell parameters [a,b,c,alpha,beta,gamma].
            labels_classification_space
            labels_classification_system (list): crystal system group.
            materials_ids (list): material ids.
        base_dir (str): base directory of repo.
        new_dir (str): where the saved data should go.

    Returns: 
        str: a key path that contains the paths of all the saved files.

    """
    features, labels_regression, labels_classification_space, labels_classification_bravais, labels_classification_system, material_id_list = get_preprocessed_returns

    print(f">> Sanity Check: printing the first element from each array\n")
    for x in get_preprocessed_returns:
        print(x[0:1])

    inp = input("<< Would you like to save your data? [y]/[n]")
    if inp != 'y':
        return print(">> No files saved. \n>> Interrupting preprocessing.")

    l = [features, labels_regression, labels_classification_space, labels_classification_bravais,
         labels_classification_system, material_id_list]

    new_folder = os.path.join(base_dir, 'preprocessed_data', f"{new_dir}")
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    length = len(material_id_list)

    filenames = [
        os.path.join(new_folder, f"{new_dir}_features_{length}.joblib"),
        os.path.join(new_folder, f"{new_dir}_regression_{length}.joblib"),
        os.path.join(new_folder, f"{new_dir}_labels_classification_space_{length}.joblib"),
        os.path.join(new_folder, f"{new_dir}_labels_classification_bravais_{length}.joblib"),
        os.path.join(new_folder, f"{new_dir}_labels_classification_system_{length}.joblib"),
        os.path.join(new_folder, f"{new_dir}_material_ids{length}.joblib")
    ]

    for i in range(len(l)):
        dump(l[i], filenames[i])

    dump(filenames, f"{new_dir}_key_path")

    print(">> All files saved to: {}".format(
        os.path.abspath(f"{new_dir}_key_path")))

    return os.path.abspath(f"{new_dir}_key_path")


def load_data(filenames_path) -> list:
    """
    Loads the data that was saved using the save_data() function.

    Args:
        filenames_path (str): path returned by save_data.

    Returns:
        list: List containing all the loaded information.
    """

    get_file_names = load(filenames_path)
    data_list = []

    for filename in get_file_names:
        print(f">> Retrieving: {filename}")
        data_list.append(load(filename))
    return data_list
