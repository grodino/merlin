from skorch import NeuralNetClassifier
from scipy.special import softmax


class PretrainedFixedNetClassifier(NeuralNetClassifier):
    """
    A classifier that uses a pre-trained neural network with fixed parameters.
    This class inherits from `NeuralNetClassifier` and overrides the `fit` and
    `initialize` methods to ensure that the parameters of the pre-trained network
    are not updated during training.
    """

    def __init__(self, module, *args, add_softmax: bool = True, **kwargs):
        """If add_softmax is True, the result of predict_proba() is passed
        through a softmax before it is returned."""
        super().__init__(module, *args, **kwargs)
        self.add_softmax = add_softmax

    def fit(self, *args, **kwargs):
        if not self.initialized_:
            self.initialize()
        return self

    def predict_proba(self, X):
        proba = super().predict_proba(X)

        if self.add_softmax:
            proba = softmax(proba, axis=1)

        return proba

    def initialize(self, *args, **kwargs):
        super().initialize()
        for p in self.module_.parameters():
            p.requires_grad = False
        return self
