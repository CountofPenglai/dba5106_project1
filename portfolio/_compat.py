"""Legacy import aliases; internal code uses canonical modules directly."""
import importlib
import sys


def alias_module(legacy_name, canonical_name):
    # Alias the module itself so patching a legacy import still reaches its globals.
    sys.modules[legacy_name] = importlib.import_module(canonical_name)
