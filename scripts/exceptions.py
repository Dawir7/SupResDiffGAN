class EvaluateFreshInitializedModelException(Exception):
    """Exception raised when a trained model file is not provided for evaluation."""

    def __init__(
        self,
        message="File with trained model not given. Please provide a trained model to evaluate.",
    ):
        self.message = message
        super().__init__(self.message)


class UnknownModeException(Exception):
    """Exception raised when an unknown mode is provided."""

    def __init__(
        self,
        message="Unknown mode provided. Please provide a valid mode: train, train-test or test.",
    ):
        self.message = message
        super().__init__(self.message)
