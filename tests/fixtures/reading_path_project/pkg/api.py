from .core import Service


def build() -> bool:
    return Service().run()
