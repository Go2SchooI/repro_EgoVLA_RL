"""Compatibility shim for benchmark scripts expecting the Isaac Lab 2.x package name."""

import importlib
import sys


_module = importlib.import_module("omni.isaac.lab")
sys.modules[__name__] = _module
