from __future__ import annotations
from typing import Callable, Dict, List, Optional


class ToolRegistry:
    """Dynamic tool registry for myvoiceai sessions."""

    def __init__(self) -> None:
        self._impls: Dict[str, Callable] = {}
        self._schemas: List[dict] = []

    def register(self, name: str, schema: dict, impl: Callable) -> None:
        """Register a tool by name with its JSON schema and callable."""
        self._impls[name] = impl
        self._schemas.append(schema)

    def lookup(self, name: str) -> Optional[Callable]:
        return self._impls.get(name)

    def schemas(self) -> List[dict]:
        return self._schemas.copy()

    def has(self, name: str) -> bool:
        return name in self._impls


_default_registry = ToolRegistry()


def get_default_registry() -> ToolRegistry:
    return _default_registry
