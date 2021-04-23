import numpy as np
from spline_visualization import layers, networks, surface


# case of random parameters
np.random.seed(11)
Ws_MLP = [
    np.random.randn(2, 16),
    np.random.randn(16, 16),
    np.random.randn(16, 16),
]
bs_MLP = [np.random.randn(16), np.random.randn(16), np.random.randn(16)]

for name, alpha in zip(["mlp_leaky", "mlp_relu", "mlp_abs"], [0.5, 0.0, -1.0]):
    network = networks.MLP(input_dim=2)
    for w, b in zip(Ws_MLP, bs_MLP):
        network.layers.append(layers.dense(network.layers[-1], 16, W=w, b=b))
        network.layers.append(
            layers.leaky_relu(network.layers[-1], alpha=alpha)
        )

    surface.pretty_plot(
        network=network,
        n_samples_x=300,
        n_samples_y=300,
        partition_figure_name=name,
    )

# specific case of zero bias and sparse W
Ws_MLP = [
    Ws_MLP,
    Ws_MLP,
    [
        w * np.random.choice([0, 1], size=np.prod(w.shape)).reshape(w.shape)
        for w in Ws_MLP
    ],
]
bs_MLP = [bs_MLP, [b * 0 for b in bs_MLP], bs_MLP]

for k, name in enumerate(["mlp_random", "mlp_biasless", "mlp_lrank"]):
    network = networks.MLP(input_dim=2)
    for w, b in zip(Ws_MLP[k], bs_MLP[k]):
        network.layers.append(layers.dense(network.layers[-1], 16, W=w, b=b))
        network.layers.append(layers.leaky_relu(network.layers[-1], alpha=0.1))

    surface.pretty_plot(
        network=network,
        n_samples_x=300,
        n_samples_y=300,
        partition_figure_name=name,
    )
