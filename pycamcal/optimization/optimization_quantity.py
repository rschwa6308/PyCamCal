from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")

class OptimizationQuantity(ABC, Generic[T]):
    @abstractmethod
    def value(self) -> T:
        "Get the current value (fixed or estimated)"

class Fixed(OptimizationQuantity[T]):
    def __init__(self, value: T):
        self._value = value
    
    def value(self):
        return self._value

class Unknown(OptimizationQuantity[T]):
    def __init__(self, initial: T):
        self._current = initial

    def value(self) -> T:
        return self._current

    def set_value(self, value: T):
        self._current = value
