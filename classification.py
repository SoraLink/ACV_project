from abc import ABC, abstractmethod


class Classification(ABC):

    @abstractmethod
    def predict(self, logit) -> bool:
        pass

    @abstractmethod
    def load(self, path):
        pass
