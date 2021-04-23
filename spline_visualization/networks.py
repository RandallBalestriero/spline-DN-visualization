from . import layers
import numpy as np
from . import utils


class MLP:
    def __init__(self, input_dim):
        self.layers = [layers.input((10, input_dim))]

    def preactivations(self, x):
        self.layers[-1].forward(x=x)
        preactivations = []
        for layer in self.layers:
            if isinstance(layer, layers.dense):
                preactivations.append([p for p in layer._value.T])
        return preactivations

    def codes(self, x):
        self.layers[-1].forward(x=x)
        codes = []
        for layer in self.layers:
            if isinstance(layer, layers.leaky_relu):
                # we reshape as matrices in case some layers were acting
                # on tensors
                codes.append(layer._mask.reshape((x.shape[0], -1)))

        # we concatenate all the codes together
        return np.concatenate(codes, 1)

    def output_mapping(self, x):
        return self.layers[-1].forward(x=x)

    def init_probing(
        self, n_samples_x=200, n_samples_y=200, extent=[-2, 2, -2, 2]
    ):
        self.paths, self.meshgrid, self.values = utils.parse_model(
            self.preactivation_mapping,
            n_samples_x=n_samples_x,
            n_samples_y=n_samples_y,
            extent=extent,
            input_code_mapping=self.code_mapping,
            n_values=10,
        )

    def init_coloring(self, colormap="Blues", alpha=1):
        import matplotlib.cm as cm

        colors = cm.get_cmap(colormap)(np.linspace(0, 1, len(self.widths)))
        self.nodes_color = [
            ["rgba" + str(tuple(color[:-1]) + (alpha,))] * width
            for color, width in zip(colors, self.widths)
        ]
        for i in range(self.widths[0]):
            self.nodes_color[0][i] = "white"

    def init_nodes_edges(self, colormap="Blues"):

        self.nodes_x = [
            np.zeros(d) + l / (len(self.widths) - 1)
            for l, d in enumerate(self.widths)
        ]

        self.nodes_y = [
            np.arange(self.widths[l]) / max(self.widths)
            + (1 - self.widths[l] / max(self.widths)) / 2
            for l, d in enumerate(self.widths)
        ]
        self.nodes_text = [
            [(1 + i, 1 + j) for j in range(self.widths[i])]
            for i in range(len(self.widths))
        ]
        edges_x = []
        edges_y = []
        for l in range(len(self.widths) - 1):
            for i in range(self.widths[l]):
                for j in range(self.widths[l + 1]):
                    edges_x.append(self.nodes_x[l][i])
                    edges_x.append(self.nodes_x[l + 1][j])
                    edges_x.append(None)

                    edges_y.append(self.nodes_y[l][i])
                    edges_y.append(self.nodes_y[l + 1][j])
                    edges_y.append(None)
        self.edges_x = edges_x
        self.edges_y = edges_y


if __name__ == "__main__":
    l0 = input((100, 800))
    l1 = dense(l0, 1000)
    l2 = sigmoid(l1)
    l3 = dense(l2, 1000)
    l4 = sigmoid(l3)
    l5 = dense(l4, 1)
    loss = MSE(l3)

    for i in range(100):
        print(
            loss.forward(
                x=np.random.randn(100, 800), y=np.random.randn(100, 1)
            ).mean()
        )
        loss.backward()
        l1.W -= 0.001 * l1._W_gradient.mean(0)
        l1.b -= 0.001 * l1._b_gradient.mean(0)
        l3.W -= 0.001 * l3._W_gradient.mean(0)
        l3.b -= 0.001 * l3._b_gradient.mean(0)
