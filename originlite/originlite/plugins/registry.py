"""Plugin registry skeleton."""
from __future__ import annotations
from typing import Callable, Dict

class PluginRegistry:
    def __init__(self):
        self.fits: Dict[str, Callable] = {}
        self.operations: Dict[str, Callable] = {}
        self.charts: Dict[str, Callable] = {}

    def register_fit(self, name: str, fn: Callable):
        self.fits[name] = fn

    def register_operation(self, name: str, fn: Callable):
        self.operations[name] = fn

    def register_chart(self, name: str, fn: Callable):
        self.charts[name] = fn

REGISTRY = PluginRegistry()
