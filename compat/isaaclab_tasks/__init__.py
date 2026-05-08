"""Compatibility shim for benchmark scripts expecting the Isaac Lab 2.x task package name."""

import importlib
import sys


_module = importlib.import_module("omni.isaac.lab_tasks")
sys.modules[__name__] = _module
