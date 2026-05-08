from importlib import import_module

_LAZY_SUBMODULES = (
    "builder",
    "dataset",
    "dataset_impl",
    "datasets_mixture",
    "simple_vila_webdataset",
)


def __getattr__(name):
    for submodule in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{submodule}")
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
