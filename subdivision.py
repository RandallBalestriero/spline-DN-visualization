import numpy as np
from spline_visualization import layers
from spline_visualization import networks
from spline_visualization import surface


np.random.seed(11)
Ws_MLP = [
    np.random.randn(2, 16),
    np.random.randn(16, 16),
    np.random.randn(16, 16),
]
bs_MLP = [np.random.randn(16), np.random.randn(16), np.random.randn(16)]
network = networks.MLP(input_dim=2)

for k, (w, b) in enumerate(zip(Ws_MLP, bs_MLP)):
    network.layers.append(layers.dense(network.layers[-1], 16, W=w, b=b))
    network.layers.append(layers.leaky_relu(network.layers[-1], alpha=0.1))

    surface.pretty_plot(
        network=network,
        n_samples_x=300,
        n_samples_y=300,
        partition_figure_name="subdivision_" + str(k + 1),
        colorize=False,
        boundary_colors=["r", "g", "b"][: k + 1],
    )


np.random.seed(1)
bWs_MLP = [
    np.random.randn(2, 16),
    np.random.randn(16, 1),
    np.random.randn(1, 16),
    np.random.randn(16, 16),
]
bbs_MLP = [
    np.random.randn(16),
    np.random.randn(1),
    np.random.randn(16),
    np.random.randn(16),
]
network = networks.MLP(input_dim=2)

for k, (w, b) in enumerate(zip(bWs_MLP, bbs_MLP)):
    network.layers.append(
        layers.dense(network.layers[-1], w.shape[1], W=w, b=b)
    )
    network.layers.append(layers.leaky_relu(network.layers[-1], alpha=0.1))

    surface.pretty_plot(
        network=network,
        n_samples_x=300,
        n_samples_y=300,
        partition_figure_name="bsubdivision_" + str(k + 1),
        colorize=False,
        boundary_colors=["r", "g", "b", "k"][: k + 1],
    )
