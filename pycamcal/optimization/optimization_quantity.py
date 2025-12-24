from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import copy

T = TypeVar("T")

class OptimizationQuantity(ABC, Generic[T]):
    @abstractmethod
    def value(self) -> T:
        "Get the current value (fixed or estimated)"

    # proxy forwarding
    def __getattr__(self, name):
        return getattr(self.value(), name)
    
    def __mul__(self, other):
        return self.value() * other
    
    def __rmul__(self, other):
        return other * self.value()

class Fixed(OptimizationQuantity[T]):
    def __init__(self, value: T):
        self._value = value
    
    def value(self):
        return self._value
    
    def __repr__(self):
        return f"Fixed({self._value})"

class Unknown(OptimizationQuantity[T]):
    def __init__(self, initial: T):
        self._current = initial

    def value(self) -> T:
        return self._current

    def set_value(self, value: T):
        self._current = value
    
    def __repr__(self):
        return f"Unknown(current={self._current})"


def VALUE(x):
    return x.value() if isinstance(x, OptimizationQuantity) else x


def RESOLVE_VALUES(obj):
    """
    Recursively make a copy of `obj` replacing any Unknown or Fixed
    instances with their current numerical values.
    """

    # If obj is an Unknown or Fixed, return its value
    if isinstance(obj, OptimizationQuantity):
        return obj.value()

    # If obj is a dict, resolve each key/value
    if isinstance(obj, dict):
        return {k: RESOLVE_VALUES(v) for k, v in obj.items()}

    # If obj is a list or tuple, resolve each element
    if isinstance(obj, list):
        return [RESOLVE_VALUES(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(RESOLVE_VALUES(x) for x in obj)

    # If obj is an object, resolve each attribute
    if hasattr(obj, "__dict__"):
        # Make a shallow copy first
        obj_copy = copy.copy(obj)
        for attr, val in vars(obj_copy).items():
            setattr(obj_copy, attr, RESOLVE_VALUES(val))
        return obj_copy

    # Fallback for other types
    return copy.deepcopy(obj)
