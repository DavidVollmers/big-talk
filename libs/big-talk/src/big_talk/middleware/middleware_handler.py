from abc import ABC, abstractmethod
from typing import Any


class MiddlewareHandler[C, R](ABC):
    @abstractmethod
    def __call__(self, ctx: C, **kwargs: Any) -> R:
        pass
