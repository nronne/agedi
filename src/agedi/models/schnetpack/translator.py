import torch

from agedi.models.translator import Translator
from agedi.data import Representation

from schnetpack.properties import n_atoms, Z, R, cell, pbc, idx_i, idx_j, offsets, idx_m, energy, forces


from agedi.data import AtomsGraph

class SchNetPackTranslator(Translator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def translate(self, batch):
        if isinstance(batch, AtomsGraph):
            out = {
                n_atoms: batch.n_atoms[:,0],
                Z: batch.x,
                R: batch.pos,
                cell: batch.cell.reshape(-1, 3, 3),
                pbc: batch.pbc,
                offsets: batch.shift_vectors,
                idx_m: batch.batch,
                idx_i: batch.edge_index[0],
                idx_j: batch.edge_index[1],
            }

            if hasattr(batch, 'energy'):
                out[energy] = batch.energy.view(-1)
            if hasattr(batch, 'forces'):
                out[forces] = batch.forces

            if batch.representation is not None:
                out['scalar_representation'] = batch.representation.scalar.squeeze(2)
                out['vector_representation'] = batch.representation.vector.permute(0, 2, 1)

            return out
        else:
            return batch
        

    def add_representation(self, batch, out):
        s, v = out['scalar_representation'], out['vector_representation']
        s = s.unsqueeze(2)
        v = torch.permute(v, (0, 2, 1))
        rep = Representation(scalar=s, vector=v)
        batch.representation = rep
        return batch

        
    def add_scores(self, batch, score):
        for k, v in score.items():
            batch[k + "_score"] = v
        return batch
        

