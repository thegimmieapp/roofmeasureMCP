"""Pure-Python stand-in for the subset of Blender's ``mathutils`` that
bpypolyskel uses (``Vector`` and ``geometry.intersect_point_line``).

The real ``mathutils`` is a C extension with no wheels for several platforms
(it failed to compile on macOS arm64 and inside the Fly builder), and it
stores components as float32. This module is double precision and needs no
compiler. roofmeasure/__init__ registers it as ``sys.modules["mathutils"]``
before bpypolyskel is imported unless a real mathutils is already loaded
(Blender).
"""
from __future__ import annotations

import math
import types
from typing import Iterable

_EPS = 1e-9


class Vector:
    __slots__ = ("_v",)

    def __init__(self, seq: Iterable[float] = (0.0, 0.0, 0.0)):
        v = [float(c) for c in seq]
        if len(v) not in (2, 3, 4):
            raise ValueError("Vector needs 2, 3 or 4 components")
        self._v = v

    # -- components -------------------------------------------------------
    @property
    def x(self) -> float:
        return self._v[0]

    @x.setter
    def x(self, val: float) -> None:
        self._v[0] = float(val)

    @property
    def y(self) -> float:
        return self._v[1]

    @y.setter
    def y(self, val: float) -> None:
        self._v[1] = float(val)

    @property
    def z(self) -> float:
        if len(self._v) < 3:
            raise AttributeError("2D Vector has no z")
        return self._v[2]

    @z.setter
    def z(self, val: float) -> None:
        if len(self._v) < 3:
            raise AttributeError("2D Vector has no z")
        self._v[2] = float(val)

    @property
    def xy(self) -> "Vector":
        return Vector(self._v[:2])

    @property
    def xyz(self) -> "Vector":
        return Vector(self._v[:3])

    def __len__(self) -> int:
        return len(self._v)

    def __iter__(self):
        return iter(self._v)

    def __getitem__(self, i):
        return self._v[i]

    def __setitem__(self, i, val):
        self._v[i] = float(val)

    def copy(self) -> "Vector":
        return Vector(self._v)

    __copy__ = copy

    def to_tuple(self, ndigits: int | None = None):
        if ndigits is None:
            return tuple(self._v)
        return tuple(round(c, ndigits) for c in self._v)

    def to_2d(self) -> "Vector":
        return Vector(self._v[:2])

    def to_3d(self) -> "Vector":
        v = list(self._v[:3])
        while len(v) < 3:
            v.append(0.0)
        return Vector(v)

    # -- arithmetic -------------------------------------------------------
    def _check(self, other: "Vector") -> None:
        if not isinstance(other, Vector):
            raise TypeError(f"Vector expected, got {type(other).__name__}")
        if len(other._v) != len(self._v):
            raise ValueError("Vector size mismatch")

    def __add__(self, other: "Vector") -> "Vector":
        self._check(other)
        return Vector(a + b for a, b in zip(self._v, other._v))

    def __sub__(self, other: "Vector") -> "Vector":
        self._check(other)
        return Vector(a - b for a, b in zip(self._v, other._v))

    def __neg__(self) -> "Vector":
        return Vector(-a for a in self._v)

    def __pos__(self) -> "Vector":
        return self.copy()

    def __mul__(self, other):
        if isinstance(other, Vector):          # mathutils: v * v is the dot product
            return self.dot(other)
        return Vector(a * float(other) for a in self._v)

    __rmul__ = __mul__

    def __truediv__(self, s: float) -> "Vector":
        s = float(s)
        return Vector(a / s for a in self._v)

    def __iadd__(self, other: "Vector") -> "Vector":
        self._check(other)
        self._v = [a + b for a, b in zip(self._v, other._v)]
        return self

    def __isub__(self, other: "Vector") -> "Vector":
        self._check(other)
        self._v = [a - b for a, b in zip(self._v, other._v)]
        return self

    def __imul__(self, s: float) -> "Vector":
        s = float(s)
        self._v = [a * s for a in self._v]
        return self

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vector) or len(other._v) != len(self._v):
            return False
        return all(abs(a - b) <= _EPS * max(1.0, abs(a), abs(b))
                   for a, b in zip(self._v, other._v))

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    __hash__ = None  # mutable, like mathutils.Vector

    # -- metrics ----------------------------------------------------------
    @property
    def length(self) -> float:
        return math.sqrt(sum(a * a for a in self._v))

    magnitude = length

    @property
    def length_squared(self) -> float:
        return sum(a * a for a in self._v)

    def dot(self, other: "Vector") -> float:
        self._check(other)
        return sum(a * b for a, b in zip(self._v, other._v))

    def cross(self, other: "Vector"):
        self._check(other)
        a, b = self._v, other._v
        if len(a) == 2:                        # mathutils: 2D cross is a scalar
            return a[0] * b[1] - a[1] * b[0]
        return Vector((a[1] * b[2] - a[2] * b[1],
                       a[2] * b[0] - a[0] * b[2],
                       a[0] * b[1] - a[1] * b[0]))

    def normalized(self) -> "Vector":
        n = self.length
        if n == 0.0:
            return Vector([0.0] * len(self._v))
        return Vector(a / n for a in self._v)

    def normalize(self) -> None:
        n = self.length
        if n != 0.0:
            self._v = [a / n for a in self._v]

    def angle(self, other: "Vector", fallback=None) -> float:
        d = self.length * other.length
        if d == 0.0:
            if fallback is not None:
                return fallback
            raise ValueError("zero-length vector")
        c = max(-1.0, min(1.0, self.dot(other) / d))
        return math.acos(c)

    def __repr__(self) -> str:
        return "Vector((" + ", ".join(f"{a:.6g}" for a in self._v) + "))"


def intersect_point_line(pt: Vector, line_p1: Vector, line_p2: Vector):
    """Closest point on the infinite line p1-p2 to pt, plus the line
    parameter (0 at p1, 1 at p2). Same contract as mathutils.geometry."""
    d = line_p2 - line_p1
    dd = d.length_squared
    if dd == 0.0:
        return line_p1.copy(), 0.0
    t = (pt - line_p1).dot(d) / dd
    return line_p1 + d * t, t


geometry = types.SimpleNamespace(intersect_point_line=intersect_point_line)

__all__ = ["Vector", "geometry", "intersect_point_line"]
