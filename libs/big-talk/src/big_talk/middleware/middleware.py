from abc import ABC, abstractmethod
from typing import Any

from .middleware_handler import MiddlewareHandler


class Middleware[C, R](ABC):
    @abstractmethod
    def __call__(self, handler: MiddlewareHandler[C, R], ctx: C, **kwargs: Any) -> R:
        pass
