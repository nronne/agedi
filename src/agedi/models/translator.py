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

    def _set_positions(self, translated: Any, pos: torch.Tensor) -> Any:
        """Replace the atomic positions inside an already-translated batch.

        Called by :meth:`translate_input` *before* the input modules run, so
        that any quantity derived from the positions (e.g. pairwise distances)
        is recomputed from *pos*.  This makes it possible to run the backbone
        on a positions tensor that carries gradients without touching
        ``batch.pos`` — whose setter clears the neighbour list and applies the
        fixed-atom mask in place, both of which would break autograd.

        The base implementation raises :class:`NotImplementedError`; subclasses
        must override it with the position key of their backend.

        Parameters
        ----------
        translated: Any
            The output of :meth:`_translate`, before input modules are applied.
        pos: torch.Tensor
            The positions to substitute, of shape ``(n_nodes, 3)``.

        Returns
        -------
        Any
            The translated batch with the positions replaced.

        Raises
        ------
        NotImplementedError
            If the subclass has not implemented this method.

        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _set_positions()"
        )

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

    def translate_input(
        self, batch: "AtomsGraph", positions: Optional[torch.Tensor] = None
    ) -> "AtomsGraph":
        """Translate the batch without injecting any stored representation.

        Unlike :meth:`__call__`, this method always skips the
        :meth:`_translate_representation` step regardless of whether
        ``batch.representation`` is set.  Use this for the *first* forward
        pass through the backbone (before the representation has been
        computed).

        Parameters
        ----------
        batch: AtomsGraph
            The batch of data to translate.
        positions: torch.Tensor, optional
            When given, these positions are substituted for ``batch.pos``
            before the input modules run, so that everything derived from the
            positions is recomputed from them.  The batch's own neighbour list
            (``edge_index`` and ``shift_vectors``) is reused as-is, so
            *positions* must correspond to the same connectivity.  Used by
            :func:`~agedi.diffusion.novelty.structure_features` to obtain a
            backbone forward pass that is differentiable with respect to the
            positions.

        Returns
        -------
        AtomsGraph
            The translated batch.
        """
        if not isinstance(batch, AtomsGraph):
            raise ValueError("Batch must be of type AtomsGraph")

        out = self._translate(batch)
        if positions is not None:
            out = self._set_positions(out, positions)
        for module in self.input_modules:
            out = module(out)
        return out

    def translate_with_representation(self, batch: "AtomsGraph") -> "AtomsGraph":
        """Translate the batch and inject the stored representation.

        Like :meth:`translate_input` but always calls
        :meth:`_translate_representation` to inject the representation
        that was previously attached via :meth:`add_representation`.
        Use this for the *second* forward pass through the backbone (after
        the representation has been computed and stored on ``batch``).

        Parameters
        ----------
        batch: AtomsGraph
            The batch of data to translate.  ``batch.representation`` must
            not be ``None`` when this method is called.

        Returns
        -------
        AtomsGraph
            The translated batch with the representation injected.
        """
        if not isinstance(batch, AtomsGraph):
            raise ValueError("Batch must be of type AtomsGraph")

        out = self._translate(batch)
        for module in self.input_modules:
            out = module(out)
        out = self._translate_representation(batch.representation, out)
        return out

    def extract_representation(self, batch: "AtomsGraph", out: Any) -> Representation:
        """Return the representation from a backbone output without storing it.

        Same conversion as :meth:`add_representation`, but leaves *batch*
        untouched.  Use this when the representation is needed for a side
        computation and must not overwrite the one the score model has already
        attached to the batch.

        Parameters
        ----------
        batch: AtomsGraph
            The original batch of data.
        out: Any
            The output of the backbone.

        Returns
        -------
        Representation
            The representation given by the model.

        """
        return self._get_representation(batch, out)

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
    

        
