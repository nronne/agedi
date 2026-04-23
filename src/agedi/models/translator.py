import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Optional
from agedi.data import Representation
from agedi.data import AtomsGraph


class Translator(ABC):
    """Base class for all translators.

    Translators are used to convert a batch of data into a format that can be used by
    the model. This is useful when the data is not in the correct format or needs to be preprocessed before being fed
    into the model.

    Parameters
    ----------
    input_modules : List[Callable]
        A list of functions that will be applied to the input data after it is translated.

    """
    def __init__(self, input_modules: Optional[List[Callable]] = None):
        """Constructor for the Translator class.

        """
        self.input_modules = input_modules if input_modules is not None else []

    def get_hparams(self) -> Dict:
        """Return hyperparameters sufficient to reconstruct this translator.

        Returns a dictionary with a ``_target_`` key (the fully-qualified class
        name) plus ``input_modules`` (each serialised with its own ``_target_``
        key where available).  Subclasses should call ``super().get_hparams()``
        and merge in their own constructor parameters.

        Returns
        -------
        dict
            Hyperparameter dictionary.
        """
        modules_hparams = []
        for m in self.input_modules:
            modules_hparams.append({
                "_target_": f"{type(m).__module__}.{type(m).__qualname__}",
            })
        return {
            "_target_": f"{type(self).__module__}.{type(self).__qualname__}",
            "input_modules": modules_hparams,
        }

    def get_representation_hparams(self, representation: Any) -> Dict:
        """Extract hyperparameters from a representation object.

        This method is called by :meth:`~agedi.models.ScoreModel.get_hparams`
        to serialise the representation (e.g. a PaiNN network) that the
        translator wraps.  The base implementation raises
        :class:`NotImplementedError`; subclasses must override it for the
        specific representation type they support.

        Parameters
        ----------
        representation : any
            The instantiated representation object.

        Returns
        -------
        dict
            Hyperparameter dictionary that can be used to reconstruct the
            representation (should contain a ``_target_`` key).

        Raises
        ------
        NotImplementedError
            If the subclass has not implemented this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_representation_hparams()"
        )
        
    @abstractmethod
    def _translate(self, batch: "AtomsGraph") -> "AtomsGraph":
        """Translate the batch of data.
        
        Abstract method that must be implemented by all subclasses.

        This method is used to translate the batch of data
        into a format that can be used by the model.

        Parameters
        ----------
        batch: AtomsGraph
            The batch of data to be translated.

        Returns
        -------
        AtomsGraph
            The translated batch of data.
        
        """
        pass

    @abstractmethod
    def _get_representation(self, batch: "AtomsGraph", out: Any) -> Representation:
        """Get the representation of the batch of data.
        
        Abstract method that must be implemented by all subclasses.

        This method is used to add the representation given by the model to the original batch of data.

        Parameters
        ----------
        batch: AtomsGraph
            The original batch of data.
        out: Any
            The output of the model.

        Returns
        -------
        Representation
            The representation given by the model.
        
        """
        pass

    @abstractmethod
    def _translate_representation(self, rep: Representation, translated_batch: Any) -> Any:
        """Translate the representation of the batch of data.

        Abstract method that must be implemented by all subclasses.

        This method is used to translate the representation given by the model back into the original batch of data.

        Parameters
        ----------
        rep: Representation
            The representation given by the model.
        translated_batch: Any
            The translated batch of data.

        Returns
        -------
        translated_batch: Any
            The translated batch of data.
        
        """
        pass

    def __call__(self, batch: "AtomsGraph") -> "AtomsGraph":
        """Call method for the Translator class.

        implementation of the __call__ method. This method is used to call the translator object as a function.

        Parameters
        ----------
        batch: AtomsGraph
            The batch of data to be translated.

        Returns
        -------
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
        """Adds the representation given by the model to the original batch of data.

        Parameters
        ----------
        batch: AtomsGraph
            The original batch of data.
        out: Any
            The output of the model.

        Returns
        -------
        AtomsGraph
            The original batch of data with the representation added.
        
        """
        batch.representation = self._get_representation(batch, out)
        return batch

    def add_scores(self, batch: "AtomsGraph", scores: Dict[str, torch.Tensor]) -> "AtomsGraph":
        """Adds the scores given by the model to the original batch of data.

        Parameters
        ----------
        batch: AtomsGraph
            The original batch of data.
        out: Dict[str, Any]
            The output of the model. Format is {head key: head predicted scores}

        Returns
        -------
        AtomsGraph
            The original batch of data with the scores added.
        
        """
        for k, v in scores.items():
            batch[k + "_score"] = v
        return batch

    
    def add_prediction(self, batch: "AtomsGraph", targets: Dict[str, torch.Tensor], type: Optional[str]=None) -> "AtomsGraph":
        """Adds the targets given by the model to the original batch of data.

        Parameters
        ----------
        batch: AtomsGraph
            The original batch of data.
        out: Dict[str, Any]
            The output of the model. Format is {head key: head predicted target}

        Returns
        -------
        AtomsGraph
            The original batch of data with the scores added.
        
        """
        for k, v in targets.items():
            if type is None:
                batch[k + "_prediction"] = v
            else:
                batch.add_batch_attr(k + "_prediction", v, type=type)
        return batch
    

        
