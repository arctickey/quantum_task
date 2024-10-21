from abc import ABC, abstractmethod
import numpy as np


class DigitClassificationInterface(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """
        Takes a 28x28x1 image and returns a single integer representing the predicted class.
        """
        pass

class CNNModel(DigitClassificationInterface):
    def __init__(self):
        pass

    def predict(self, image: np.ndarray) -> int:
        raise NotImplementedError("CNN prediction not implemented.")


class RandomForestModel(DigitClassificationInterface):
    def __init__(self):
        pass

    def predict(self, image: np.ndarray) -> int:
        flat_image = image.flatten()
        if flat_image.shape != (784,):
            raise ValueError("Input must flatten to a 1D array of length 784.")
        raise NotImplementedError("Random Forest prediction not implemented.")


class RandomClassifier(DigitClassificationInterface):
    def __init__(self):
        pass

    def predict(self, image: np.ndarray) -> int:
        center_crop = image[9:19, 9:19, 0]
        if center_crop.shape != (10, 10):
            raise ValueError("Center crop must be 10x10.")
        return np.random.randint(0, 10)


class DigitClassifier:
    def __init__(self, algorithm: str):
        """
        Initializes the classifier with the specified algorithm.

        Args:
        algorithm (str): The name of the algorithm ('cnn', 'rf', 'rand').
        """
        self.models = {
            'cnn': CNNModel(),
            'rf': RandomForestModel(),
            'rand': RandomClassifier()
        }
        if algorithm not in self.models:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self.model = self.models[algorithm]

    def predict(self, image: np.ndarray) -> int:
        """
        Takes a 28x28x1 image and returns the predicted digit.

        Args:
        image (np.ndarray): The input image (28x28x1).

        Returns:
        int: The predicted digit.
        """
        if image.shape != (28, 28, 1):
            raise ValueError("Input must be a 28x28x1 image.")
        return self.model.predict(image)
