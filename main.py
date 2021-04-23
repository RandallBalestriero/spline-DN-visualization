import numpy as np
import layers
import networks
import surface


#np.random.seed(8)
#surface.pretty_onelayer_partition(
#    np.random.rand(3, 2) * 4 - 2,
#    np.random.randn(3) * 0.5,
#    -0.01,
#    name="one_layer_PD.pdf",
#)
#K = 6
#np.random.seed(1)
#surface.pretty_onelayer_partition(
#    np.random.rand(K, 2) * 1 - 0.5,
#    np.random.randn(K) * 0.0,
#    -0.1,
#    with_power_diagram=True,
#    name="one_layer_biasless.pdf",
#)
#
#np.random.seed(1)
#W = np.random.randn(K, 2) * 0.4
#for k in range(K):
#    W[k, int(np.random.rand() < 0.5)] = 0
#surface.pretty_onelayer_partition(
#    W,
#    np.random.rand(K) * 2 - 1,
#    -0.1,
#    with_power_diagram=True,
#    name="one_layer_l0.pdf",
#)
#
#np.random.seed(2)
#theta = np.linspace(-np.pi, np.pi, 100)
#W = np.stack([np.cos(theta), np.sin(theta)], 1)
#surface.pretty_onelayer_partition(
#    W,
#    np.ones(100) * 0.5,
#    -0.01,
#    with_power_diagram=False,
#    name="one_layer_circle.pdf",
#)
np.random.seed(11)
Ws_MLP = [np.random.randn(2,16), np.random.randn(16,16), np.random.randn(16,16)]
bs_MLP = [np.random.randn(16), np.random.randn(16), np.random.randn(16)]

#for name, alpha in zip(["mlp_leaky", "mlp_relu", "mlp_abs"], [0.5, 0., -1.]):
#    network = networks.MLP(input_dim=2)
#    for w,b in zip(Ws_MLP, bs_MLP):
#        network.layers.append(layers.dense(network.layers[-1], 16, W=w, b=b))
#        network.layers.append(layers.leaky_relu(network.layers[-1], alpha=alpha))
#    
#    surface.pretty_plot(
#            network=network,
#            n_samples_x=300,
#            n_samples_y=300,
#            partition_figure_name=name
#    )

Ws_MLP = [Ws_MLP, Ws_MLP, [w*np.random.choice([0, 1], size=np.prod(w.shape)).reshape(w.shape) for w in Ws_MLP]]
bs_MLP = [bs_MLP, [b*0 for b in bs_MLP], bs_MLP]

#for k, name in enumerate(["mlp_random", "mlp_biasless", "mlp_lrank"]):
#    network = networks.MLP(input_dim=2)
#    for w,b in zip(Ws_MLP[k], bs_MLP[k]):
#        network.layers.append(layers.dense(network.layers[-1], 16, W=w, b=b))
#        network.layers.append(layers.leaky_relu(network.layers[-1], alpha=0.1))
#    
#    surface.pretty_plot(
#            network=network,
#            n_samples_x=300,
#            n_samples_y=300,
#            partition_figure_name=name
#    )

network = networks.MLP(input_dim=2)
for k, (w,b) in enumerate(zip(Ws_MLP[0], bs_MLP[0])):
    network.layers.append(layers.dense(network.layers[-1], 16, W=w, b=b))
    network.layers.append(layers.leaky_relu(network.layers[-1], alpha=0.1))

    surface.pretty_plot(
        network=network,
        n_samples_x=300,
        n_samples_y=300,
        partition_figure_name="subdivision_"+str(k + 1),
        colorize=False,
        boundary_colors=["r", "g", "b"][:k+1]
    )


np.random.seed(1)
bWs_MLP = [np.random.randn(2,16), np.random.randn(16,1), np.random.randn(1,16), np.random.randn(16,16)]
bbs_MLP = [np.random.randn(16), np.random.randn(1), np.random.randn(16), np.random.randn(16)]

network = networks.MLP(input_dim=2)
for k, (w,b) in enumerate(zip(bWs_MLP, bbs_MLP)):
    network.layers.append(layers.dense(network.layers[-1], w.shape[1], W=w, b=b))
    network.layers.append(layers.leaky_relu(network.layers[-1], alpha=0.1))

    surface.pretty_plot(
        network=network,
        n_samples_x=300,
        n_samples_y=300,
        partition_figure_name="bsubdivision_"+str(k + 1),
        colorize=False,
        boundary_colors=["r", "g", "b", "k"][:k+1]
    )













adf
np.random.seed(11)
model = MLP([layers.input((10, 2))], [8, 8], alpha=0.3)
model.layers.append(layers.dense(model.layers[-1], 2))
from scipy.stats import multivariate_normal

var = multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
surface.pretty_plot(
    model.preactivation_mapping,
    model.output_mapping,
    model.code_mapping,
    n_samples_x=200,
    n_samples_y=200,
    color_mapping=var.pdf,
    name_input_space="input_space_2d_g.pdf",
    name_output_space="output_space_2d_g.pdf",
)


# np.random.seed(12)
# model = MLP([layers.input((10, 2))], [8, 8], alpha=0.3)
# model.layers.append(layers.dense(model.layers[-1], 3))

# surface.pretty_plot(
#     model.preactivation_mapping,
#     model.output_mapping,
#     model.code_mapping,
#     n_samples_x=200,
#     n_samples_y=200,
#     name_input_space="input_space_3d.pdf",
#     name_output_space="output_space_3d.pdf",
# )
