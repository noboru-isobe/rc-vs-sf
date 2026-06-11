from .base import Trajectory
from .double_pendulum import double_pendulum
from .langevin import langevin
from .linear import diagonal_lds, gaussian_lds, permutation_lds
from .lorenz import lorenz
from .mackey_glass import mackey_glass
from .narma import narma10

SYSTEMS = {
    "gaussian_lds": gaussian_lds,
    "permutation_lds": permutation_lds,
    "diagonal_lds": diagonal_lds,
    "lorenz": lorenz,
    "lorenz_partial": lambda seed, **kw: lorenz(seed, partial=True, **kw),
    "double_pendulum": double_pendulum,
    "double_pendulum_partial": lambda seed, **kw: double_pendulum(seed, partial=True, **kw),
    "langevin": langevin,
    "narma10": narma10,
    "mackey_glass": mackey_glass,
}

__all__ = ["Trajectory", "SYSTEMS", *SYSTEMS.keys()]
