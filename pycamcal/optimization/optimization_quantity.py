from abc import ABC, abstractmethod
from typing import Generic, TypeVar

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

