"""The single operating-system dispatch key and the helper that spreads one implementation over several systems."""

import platform

from patos import value_dispatch


def system_key() -> str:
    return platform.system()


def otherwise[**P, R](dispatcher: value_dispatch[P, R], *systems: str) -> None:
    """Serve each of `systems` with the dispatcher's own fallback, since krn refuses a key nobody registered."""

    def fallback(*args: P.args, **kwargs: P.kwargs) -> R:
        return dispatcher(*args, **kwargs)

    for system in systems:
        dispatcher.register(system)(fallback)
