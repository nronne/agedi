from .base import Noiser
from .sde import SDENoiser
from .pos import PositionsNoiser, Positions, CellPositions, ConfinedCellPositions
from .types import Types, TypesNoiser

# ---------------------------------------------------------------------------
# Built-in noiser registry entries
# ---------------------------------------------------------------------------
# Register the built-in noisers so they can be referenced by name in
# :func:`~agedi.functional.create_diffusion`.  The factory callable receives
# ``sde`` as a keyword argument; noisers that do not use an SDE (like
# :class:`Types`) can simply ignore it.

# CamelCase names (primary)
Noiser.register("Positions", lambda sde: Positions(sde=sde))
Noiser.register("CellPositions", lambda sde: CellPositions(sde=sde))
Noiser.register("ConfinedCellPositions", lambda sde: ConfinedCellPositions(sde=sde))
Noiser.register("Types", lambda sde: Types())

# snake_case aliases for backwards compatibility
Noiser.register("positions", lambda sde: Positions(sde=sde))
Noiser.register("cell_positions", lambda sde: CellPositions(sde=sde))
Noiser.register("confined_cell_positions", lambda sde: ConfinedCellPositions(sde=sde))
Noiser.register("types", lambda sde: Types())

__all__ = [
    "Noiser",
    "SDENoiser",
    "PositionsNoiser",
    "Positions",
    "CellPositions",
    "ConfinedCellPositions",
    "Types",
    "TypesNoiser",
]
