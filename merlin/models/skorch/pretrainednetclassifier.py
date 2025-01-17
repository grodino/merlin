from skorch import NeuralNetClassifier

class PretrainedFixedNetClassifier(NeuralNetClassifier):
    """
    A classifier that uses a pre-trained neural network with fixed parameters.
    This class inherits from `NeuralNetClassifier` and overrides the `fit` and 
    `initialize` methods to ensure that the parameters of the pre-trained network 
    are not updated during training.
    """
    def fit(self, *args, **kwargs):
        if not self.initialized_:
            self.initialize()
        return self
            
    def initialize(self, *args, **kwargs):
        super().initialize()
        for p in self.module_.parameters():
            p.requires_grad = False
        return self