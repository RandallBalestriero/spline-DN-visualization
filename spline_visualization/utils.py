import numpy as np
import matplotlib.pyplot as plt


def zero_set_paths(feature_maps, meshgrid):
    """
    Parameters:
    -----------

    feature_maps: nested list of matrices, same shape as flattened X or flattened Y
        each vector represents the output mapping of a specific
        neuron to compute the zero set from

    meshgrid: couple of matrices (X, Y)
        the sampling of the input space used to generated the feature_maps
        vectors

    Returns:
    --------

    zero_set_paths: nested list of list
        with same hierarchy than feature_maps, containing list of paths for each neuron
    """

    if type(feature_maps) == list:
        return [zero_set_paths(p, meshgrid) for p in feature_maps]

    paths = plt.contour(
        meshgrid[0], meshgrid[1], feature_maps.reshape(meshgrid[0].shape), [0]
    )
    paths = paths.collections[0].get_paths()
    plt.close()

    vertices = []
    for path in paths:
        clean_vertices = path.cleaned(simplify=True).vertices
        vertices.append(clean_vertices[:-1])

    return vertices


def parse_model(
    mapping,
    n_samples_x=100,
    n_samples_y=100,
    extent=[-3, 3, -3, 3],
    meshgrid=None,
    input_code_mapping=None,
    n_values=10,
    need_input=True,
):

    print("--- PARSING")

    if not meshgrid:
        grid_x = np.linspace(extent[0], extent[1], n_samples_x)
        grid_y = np.linspace(extent[2], extent[3], n_samples_y)
        meshgrid = np.meshgrid(grid_x, grid_y)

    meshgrid_flat = np.hstack([grid.reshape((-1, 1)) for grid in meshgrid])

    if need_input:
        feature_maps = mapping(meshgrid_flat)
    else:
        feature_maps = mapping()

    paths = get_zero_set_paths(feature_maps, meshgrid)

    if input_code_mapping is not None:
        codes = input_code_mapping(meshgrid_flat)
        unique_codes = set(map(tuple, codes))
        cmap_sampling = np.linspace(0, 1, min(n_values, len(unique_codes)))
        values = cmap_sampling[np.arange(len(unique_codes)) % n_values]
        code_to_value = {a: t for a, t in zip(unique_codes, values)}

        values = np.array([code_to_value[tuple(code)] for code in codes])

    else:
        values = []

    return paths, meshgrid, values
