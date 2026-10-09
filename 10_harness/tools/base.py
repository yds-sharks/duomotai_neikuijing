#!/usr/bin/env python3
"""Tool base class and registry: RAG capabilities exposed to the central brain."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:  # avoid circular import at runtime
    from runtime.session import AgentSession


@dataclass
class ToolResult:
    ok: bool
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    n_new: int = 0
    message: str = ""
    finish: bool = False  # True when the tool ends the episode (submit_answer)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "candidates": self.candidates,
            "n_new": self.n_new,
            "message": self.message,
            "finish": self.finish,
        }


class Tool(ABC):
    """A capability the brain may call once per round."""

    name: str = ""
    description: str = ""
    args_schema: str = ""

    @abstractmethod
    def run(self, session: "AgentSession", args: Dict[str, Any]) -> ToolResult: ...

    def spec(self) -> Dict[str, str]:
        return {"name": self.name, "description": self.description, "args": self.args_schema}


class ToolRegistry:
    def __init__(self, tools: List[Tool]):
        self._tools: Dict[str, Tool] = {}
        for t in tools:
            if not t.name:
                raise ValueError(f"tool {type(t).__name__} has empty name")
            if t.name in self._tools:
                raise ValueError(f"duplicate tool name: {t.name}")
            self._tools[t.name] = t

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name!r}; available: {sorted(self._tools)}")
        return self._tools[name]

    def specs(self) -> List[Dict[str, str]]:
        return [t.spec() for t in self._tools.values()]

    def names(self) -> List[str]:
        return sorted(self._tools)
