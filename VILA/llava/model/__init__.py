from importlib import import_module

__all__ = ["LlavaLlamaConfig", "LlavaLlamaModel"]


def __getattr__(name):
    if name in __all__:
        module = import_module(".language_model.llava_llama", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
