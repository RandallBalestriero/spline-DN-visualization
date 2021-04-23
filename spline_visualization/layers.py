from scipy.special import expit, softmax
import numpy as np


class Layer:
    def forward(self, **kwargs):
        pass

    def backward(self, input_grad):
        pass


class NonlinearLayer(Layer):
    pass


class LossLayer(Layer):
    pass


class input(Layer):
    def __init__(self, input_shape, name="x"):
        self.output_shape = input_shape
        self.name = name

    def forward(self, **kwargs):
        self._value = kwargs[self.name]
        return self._value


class sigmoid(NonlinearLayer, Layer):
    def __init__(self, previous_layer):
        self.output_shape = previous_layer.output_shape
        self.previous_layer = previous_layer

    def forward(self, **kwargs):
        self._value = expit(self.previous_layer.forward(**kwargs))
        return self._value

    def backward(self, input_grad):
        derivative = input_grad * self._value * (1 - self._value)
        self.previous_layer.backward(derivative)


class leaky_relu(NonlinearLayer, Layer):
    def __init__(self, previous_layer, alpha=0.01, beta=1):
        self.output_shape = previous_layer.output_shape
        self.previous_layer = previous_layer
        self.alpha = alpha
        self.beta = beta

    def forward(self, **kwargs):
        input = self.previous_layer.forward(**kwargs)
        if self.beta == 1:
            self._mask = (input > 0).astype("float32")
            self._value = np.where(self._mask, input, self.alpha * input)
            return self._value
        else:
            coeff = self.beta / (1 - self.beta)
            denom = np.exp(self.alpha * coeff * input) + np.exp(coeff * input)
            left = (
                self.alpha * input * np.exp(self.alpha * coeff * input) / denom
            )
            right = input * np.exp(coeff * input) / denom
            self._mask = None
            self._value = left + right
            return self._value

    def backward(self, input_grad):
        self.previous_layer.backward(
            self.mask * derivative + self.alpha * (1 - self.mask) * input_grad
        )


class relu(leaky_relu):
    def __init__(self, previous_layer):
        self.output_shape = previous_layer.output_shape
        self.previous_layer = previous_layer
        self.alpha = 0


class abs(leaky_relu):
    def __init__(self, previous_layer):
        self.output_shape = previous_layer.output_shape
        self.previous_layer = previous_layer
        self.alpha = -1


class dense(Layer):
    def __init__(self, previous_layer, n_units, W=None, b=None):
        assert len(previous_layer.output_shape) == 2
        self.output_shape = (previous_layer.output_shape[0], n_units)
        self.previous_layer = previous_layer

        if W is None:
            self.W = np.random.randn(
                previous_layer.output_shape[1], n_units
            ) / np.sqrt(previous_layer.output_shape[1])
        else:
            self.W = W

        if b is None:
            self.b = np.random.randn(n_units) / np.sqrt(
                previous_layer.output_shape[1]
            )
        else:
            self.b = b

    def forward(self, **kwargs):
        self._value = (
            self.previous_layer.forward(**kwargs).dot(self.W) + self.b
        )
        return self._value

    def backward(self, input_grad):
        self._W_gradient = np.einsum(
            "nk,nd->ndk", input_grad, self.previous_layer._value
        )
        self._b_gradient = input_grad
        self.previous_layer.backward(input_grad.dot(self.W.T))


def Conv2D(x, W, mode="valid"):
    y = []
    for Wk in self.W:
        y.append(
            sum([convolve(xc, Wkc, mode=self.mode) for xc, Wkc in zip(x, Wk)])
        )
    return np.array(y)


class convolution(Layer):
    def __init__(self, previous_layer, n_filters, filters_shape, mode="valid"):
        assert len(previous_layer.output_shape) == 2
        input_shape = previous_layer.output_shape
        self.output_shape = (input_shape[0], n_filters)
        self.mode = mode
        if mode == "valid":
            self.output_shape += (
                input_shape[2] - filters_shape[0] + 1,
                input_shape[3] - filters_shape[1] + 1,
            )
        elif mode == "same":
            self.output_shape += input_shape[2:]

        self.previous_layer = previous_layer

        self.W = np.random.randn(
            n_filters, input_shape[1], *filters_shape
        ) / np.sqrt(np.prod(filters_shape) * input_shape[1])
        self.b = np.random.randn(n_filters)

    def forward(self, **kwargs):
        outputs = []
        for x in self.previous_layer.forward(**kwargs):
            y = []

        self._value = np.array(outputs)
        return self._value

    def backward(self, input_grad):
        self._W_gradient = np.einsum(
            "nk,nd->ndk", input_grad, self.previous_layer._value
        )
        self._b_gradient = input_grad

        outputs = []
        for x in input_grad:
            y = []
            for Wk in self.W:
                y.append(self._tensor_conv(x, Wk[::-1, ::-1]))
            outputs.append(y)

        self.previous_layer.backward(np.array(outputs))


class softmax(NonlinearLayer, Layer):
    def __init__(self, previous_layer):
        self.output_shape = previous_layer.output_shape
        self.previous_layer = previous_layer

    def forward(self, **kwargs):
        self._value = softmax(self.previous_layer.forward(**kwargs))
        return self._value

    def backward(self, input_grad):
        # First we create for each example feature vector, it's outer product with itself
        # ( p1^2  p1*p2  p1*p3 .... )
        # ( p2*p1 p2^2   p2*p3 .... )
        # ( ...                     )
        tensor1 = np.einsum(
            "ij,ik->ijk", self._value, self._value
        )  # (m, n, n)
        # Second we need to create an (n,n) identity of the feature vector
        # ( p1  0  0  ...  )
        # ( 0   p2 0  ...  )
        # ( ...            )
        tensor2 = np.einsum(
            "ij,jk->ijk", self._value, np.eye(n, n)
        )  # (m, n, n)
        # Then we need to subtract the first tensor from the second
        # ( p1 - p1^2   -p1*p2   -p1*p3  ... )
        # ( -p1*p2     p2 - p2^2   -p2*p3 ...)
        # ( ...                              )
        dSoftmax = tensor2 - tensor1
        # Finally, we multiply the dSoftmax (da/dz) by da (dL/da) to get the gradient w.r.t. Z
        derivative = np.einsum("ijk,ik->ij", dSoftmax, input_grad)  # (m, n)
        self.previous_layer.backward(derivative)


class MSE(LossLayer, Layer):
    def __init__(self, previous_layer):
        self.output_shape = ()
        self.previous_layer = previous_layer

    def forward(self, **kwargs):
        assert "y" in kwargs
        self._diff = self.previous_layer.forward(**kwargs) - kwargs["y"]
        axes = range(1, len(self.previous_layer.output_shape))
        self._value = ((self._diff) ** 2).mean(tuple(axes))
        return self._value

    def backward(self):
        self.previous_layer.backward(self._diff)


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
