from abc import ABC, abstractmethod
from agedi.data import Representation


class Translator(ABC):
    """
    Base class for all translators. Translators are used to convert a batch of data into a format that can be used by
    the model. This is useful when the data is not in the correct format or needs to be preprocessed before being fed
    into the model.

    """
    def __call__(self, batch: "AtomsGraph") -> "AtomsGraph":
        """
        implementation of the __call__ method. This method is used to call the translator object as a function.

        Args:

        batch: AtomsGraph
            The batch of data to be translated.

        Returns:

        AtomsGraph
            The translated batch of data.
        """
        return self.translate(batch)
        
    @abstractmethod
    def translate(self, batch: "AtomsGraph") -> "AtomsGraph":
        """
        Abstract method that must be implemented by all subclasses. This method is used to translate the batch of data
        into a format that can be used by the model.

        Args:

        batch: AtomsGraph
            The batch of data to be translated.

        Returns:

        AtomsGraph
            The translated batch of data.
        """
        pass

    @abstractmethod
    def add_representation(self, batch, out):
        return batch

    @abstractmethod
    def add_scores(self, batch, out):
        return batch
        
