import torch
from abc import ABC, abstractmethod
from typing import List, Callable, Any, Dict
from agedi.data import Representation
from agedi.data import AtomsGraph


class Translator(ABC):
    """
    Base class for all translators. Translators are used to convert a batch of data into a format that can be used by
    the model. This is useful when the data is not in the correct format or needs to be preprocessed before being fed
    into the model.

    """
    def __init__(self, input_modules: List[Callable]=[]):
        """
        Constructor for the Translator class.

        Args:

        input_modules: List[Callable]
            A list of functions that will be applied to the input data after it is translated.
        """
        self.input_modules = input_modules
        
    @abstractmethod
    def _translate(self, batch: "AtomsGraph") -> "AtomsGraph":
        """
        Abstract method that must be implemented by all subclasses.

        This method is used to translate the batch of data
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
    def _get_representation(self, batch: "AtomsGraph", out: Any) -> Representation:
        """
        Abstract method that must be implemented by all subclasses.

        This method is used to add the representation given by the model to the original batch of data.

        Args:

        batch: AtomsGraph
            The original batch of data.

        out: Any
            The output of the model.

        Returns:

        Representation
            The representation given by the model.
        
        """
        pass

    @abstractmethod
    def _translate_representation(self, rep: Representation, translated_batch: Any) -> Any:
        """
        Abstract method that must be implemented by all subclasses.

        This method is used to translate the representation given by the model back into the original batch of data.

        Args:

        rep: Representation
            The representation given by the model.

        translated_batch: Any
            The translated batch of data.

        Returns:

        translated_batch: Any
            The translated batch of data.
        
        """
        pass

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
        if not isinstance(batch, AtomsGraph):
            raise ValueError("Batch must be of type AtomsGraph")
        
        out = self._translate(batch)
        for module in self.input_modules:
            out = module(out)

        if batch.representation is not None:
            out = self._translate_representation(batch.representation, out)
            
        return out

    def add_representation(self, batch: "AtomsGraph", out: Any) -> "AtomsGraph":
        """
        Adds the representation given by the model to the original batch of data.

        Args:

        batch: AtomsGraph
            The original batch of data.

        out: Any
            The output of the model.

        Returns:

        AtomsGraph
        
        """
        batch.representation = self._get_representation(batch, out)
        return batch

    def add_scores(self, batch: "AtomsGraph", scores: Dict[str, torch.Tensor]) -> "AtomsGraph":
        """
        Adds the scores given by the model to the original batch of data.

        Args:

        batch: AtomsGraph
            The original batch of data.

        out: Dict[str, Any]
            The output of the model. Format is {head key: head predicted scores}


        Returns:

        AtomsGraph
        
        """
        for k, v in scores.items():
            batch[k + "_score"] = v
        return batch

        
