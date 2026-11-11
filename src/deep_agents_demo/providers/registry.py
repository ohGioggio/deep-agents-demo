"""A registry of named providers, filled with a decorator (Registry + Strategy).

One instance per family of swappable implementations; a setting picks one by name::

    SANDBOXES = ProviderRegistry("sandbox provider")

    @SANDBOXES.register("openshell")
    def openshell(settings, data, stack, notify): ...

    factory = SANDBOXES.get(settings.provider)

Adding an implementation is one decorated function; nothing else changes.
"""
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T", bound=Callable)


class ProviderRegistry(Generic[T]):
    def __init__(self, kind: str = "provider"):
        self.kind = kind
        self._providers: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator: register the function under ``name`` and return it unchanged."""
        def add(provider: T) -> T:
            if name in self._providers:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._providers[name] = provider
            return provider
        return add

    def get(self, name: str) -> T:
        try:
            return self._providers[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} {name!r} (known: {', '.join(self.names())})") from None

    def names(self) -> tuple[str, ...]:
        """Registered names, in registration order."""
        return tuple(self._providers)
