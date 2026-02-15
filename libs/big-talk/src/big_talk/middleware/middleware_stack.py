from typing import Callable, Any

from .middleware import Middleware
from .middleware_handler import MiddlewareHandler


class _MiddlewareWrapper[C, R](MiddlewareHandler[C, R]):
    def __init__(self, mw: Middleware[C, R], next_handler: MiddlewareHandler[C, R]):
        self._mw = mw
        self._next_handler = next_handler

    def __call__(self, ctx: C, **kwargs: Any) -> R:
        return self._mw(self._next_handler, ctx, **kwargs)


class _CallableMiddlewareAdapter[C, R](Middleware[C, R]):
    def __init__(self, func: Callable[[MiddlewareHandler[C, R], C, Any], R]):
        self._func = func

    def __call__(self, handler: MiddlewareHandler[C, R], ctx: C, **kwargs: Any) -> R:
        return self._func(handler, ctx, **kwargs)


class MiddlewareStack[C, R]:
    def __init__(self, base_handler: MiddlewareHandler[C, R]):
        self._middleware: list[Middleware[C, R]] = []
        self._base_handler = base_handler

    def use(self, mw: Middleware[C, R] | Callable) -> None:
        if not isinstance(mw, Middleware):
            mw = _CallableMiddlewareAdapter(mw)
        self._middleware.append(mw)

    def build(self) -> MiddlewareHandler[C, R]:
        handler = self._base_handler
        for mw in reversed(self._middleware):
            handler = _MiddlewareWrapper(mw, handler)
        return handler
