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
Noiser.register("Positions", lambda sde, **kw: Positions(sde=sde, **kw))
Noiser.register("CellPositions", lambda sde, **kw: CellPositions(sde=sde, **kw))
Noiser.register("ConfinedCellPositions", lambda sde, **kw: ConfinedCellPositions(sde=sde, **kw))
Noiser.register("Types", lambda sde, **kw: Types())

# snake_case aliases for backwards compatibility
Noiser.register("positions", lambda sde, **kw: Positions(sde=sde, **kw))
Noiser.register("cell_positions", lambda sde, **kw: CellPositions(sde=sde, **kw))
Noiser.register("confined_cell_positions", lambda sde, **kw: ConfinedCellPositions(sde=sde, **kw))
Noiser.register("types", lambda sde, **kw: Types())

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
