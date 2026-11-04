"""Subclass sets, handler order, ids, change detection and exchange of the class table, over random hierarchies."""

import builtins
import json

import pytest
from hypothesis import given, reject, settings
from hypothesis import strategies as st

from krn.exceptions import table as table_module
from krn.exceptions.table import BUILTINS, ClassChangedError, ErrorClassTable

LEAVES = (ValueError, KeyError, RuntimeError, LookupError, ArithmeticError, Exception, BaseException, KeyboardInterrupt)


@st.composite
def hierarchies(draw: st.DrawFn, size: int = 10) -> tuple[type[BaseException], ...]:
    """Classes built with type(): each new one derives from one or two earlier classes or a leaf builtin."""
    created: list[type[BaseException]] = []
    for index in range(draw(st.integers(0, size))):
        pool = [*LEAVES, *created]
        bases = tuple(draw(st.lists(st.sampled_from(pool), min_size=1, max_size=2, unique=True)))
        try:
            created.append(type(f"Made{index}", bases, {}))
        except TypeError:
            reject()
    return tuple(created)


def universe(made: tuple[type[BaseException], ...]) -> tuple[type[BaseException], ...]:
    return (*BUILTINS, *made)


@settings(max_examples=150, deadline=None)
@given(hierarchies())
def test_matches_is_issubclass_among_registered_classes(made: tuple[type[BaseException], ...]) -> None:
    table = ErrorClassTable()
    classes = universe(made)
    for cls in made:
        table.register(cls)
    for raised in classes:
        for handler in classes:
            assert table.matches(table.register(raised), handler) == issubclass(raised, handler)


@settings(max_examples=150, deadline=None)
@given(hierarchies())
def test_handler_ids_are_sorted_subclass_sets(made: tuple[type[BaseException], ...]) -> None:
    table = ErrorClassTable()
    classes = universe(made)
    for cls in made:
        table.register(cls)
    for handler in classes:
        expected = sorted(table.register(cls) for cls in classes if issubclass(cls, handler))
        assert table.handler_ids(handler) == tuple(expected)


@settings(max_examples=150, deadline=None)
@given(hierarchies(), st.data())
def test_first_matching_handler_wins_as_in_python(made: tuple[type[BaseException], ...], data: st.DataObject) -> None:
    table = ErrorClassTable()
    classes = universe(made)
    raised = data.draw(st.sampled_from(classes))
    clauses = data.draw(st.lists(st.sampled_from(classes), min_size=1, max_size=6))
    python = next((index for index, handler in enumerate(clauses) if issubclass(raised, handler)), None)
    ours = next(
        (index for index, handler in enumerate(clauses) if table.matches(table.register(raised), handler)), None
    )
    assert ours == python


@settings(max_examples=100, deadline=None)
@given(hierarchies())
def test_ids_depend_only_on_registration_order(made: tuple[type[BaseException], ...]) -> None:
    first, second = ErrorClassTable(), ErrorClassTable()
    assert [first.register(cls) for cls in made] == [second.register(cls) for cls in made]
    assert all(first.cls_of(first.register(cls)) is cls for cls in made)


def test_builtins_have_fixed_ids_in_every_program() -> None:
    table = ErrorClassTable()
    assert [table.register(cls) for cls in BUILTINS] == list(range(len(BUILTINS)))
    assert table.register(ValueError) == BUILTINS.index(ValueError)


def test_registering_a_class_registers_its_bases_first() -> None:
    class Outer(KeyError):
        pass

    class Inner(Outer, RuntimeError):
        pass

    table = ErrorClassTable()
    identifier = table.register(Inner)
    assert identifier == len(BUILTINS) + 1
    assert table.cls_of(identifier - 1) is Outer and table.cls_of(identifier) is Inner
    assert table.matches(identifier, KeyError) and table.matches(identifier, RuntimeError)
    assert not table.matches(table.register(KeyError), Inner)


def test_only_exception_classes_register() -> None:
    untyped = vars(builtins)["int"]
    with pytest.raises(TypeError, match="not an exception class"):
        ErrorClassTable().register(untyped)


def test_ids_beyond_the_device_record_limit_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(table_module, "MAX_ID", len(BUILTINS) - 1)
    table = ErrorClassTable()

    class Overflow(ValueError):
        pass

    with pytest.raises(OverflowError, match="Overflow"):
        table.register(Overflow)


def test_a_changed_base_raises_instead_of_answering_from_the_old_hierarchy() -> None:
    class Parent(RuntimeError):
        pass

    class Child(Parent):
        pass

    table = ErrorClassTable()
    identifier = table.register(Child)
    assert table.matches(identifier, RuntimeError)
    Parent.__bases__ = (KeyError,)
    with pytest.raises(ClassChangedError, match="Child changed"):
        table.matches(identifier, RuntimeError)
    with pytest.raises(ClassChangedError, match="Parent changed"):
        table.register(Parent)


def test_a_replaced_constructor_raises_on_the_next_query() -> None:
    class Replaced(RuntimeError):
        def __init__(self, limit: int) -> None:
            super().__init__(limit)

    table = ErrorClassTable()
    identifier = table.register(Replaced)
    assert table.device_reason(Replaced) is None
    Replaced.__init__ = lambda self, limit: None
    with pytest.raises(ClassChangedError, match="Replaced changed"):
        table.matches(identifier, RuntimeError)
    assert table.cls_of(identifier) is Replaced


def test_reading_a_class_annotations_is_not_a_change() -> None:
    class Annotated(RuntimeError):
        limit: int

    table = ErrorClassTable()
    identifier = table.register(Annotated)
    assert Annotated.__annotations__ == {"limit": int}
    assert table.matches(identifier, RuntimeError)


def test_an_unresolved_annotation_does_not_stop_registration() -> None:
    namespace: dict[str, type[BaseException]] = {}
    exec(
        "class Deferred(RuntimeError):\n    def method(self, value: NotDefinedAnywhere) -> None:\n        pass\n",
        namespace,
    )
    table = ErrorClassTable()
    assert table.matches(table.register(namespace["Deferred"]), RuntimeError)


def test_changed_function_code_is_a_change_too() -> None:
    class Patched(RuntimeError):
        def __init__(self, limit: int) -> None:
            super().__init__(limit)

        def replacement(self, limit: int) -> None:
            super().__init__(limit + 1)

    table = ErrorClassTable()
    identifier = table.register(Patched)
    Patched.__init__.__code__ = Patched.replacement.__code__
    with pytest.raises(ClassChangedError):
        table.matches(identifier, RuntimeError)


def test_a_change_in_an_unrelated_handler_is_found_by_handler_ids() -> None:
    class Stable(RuntimeError):
        pass

    class Other(RuntimeError):
        pass

    table = ErrorClassTable()
    table.register(Stable)
    table.register(Other)
    Other.__bases__ = (KeyError,)
    with pytest.raises(ClassChangedError):
        table.handler_ids(RuntimeError)


def test_device_reason_names_the_construct_that_is_not_device_safe() -> None:
    class Loose(RuntimeError):
        def __init__(self, *args: int, **kwargs: int) -> None:
            super().__init__(*args)

    class Unsafe(Loose):
        def __init__(self, limit: int) -> None:
            super().__init__(limit=limit)

    table = ErrorClassTable()
    reason = table.device_reason(Unsafe)
    assert reason is not None and "super().__init__(limit=limit)" in reason.text
    assert table.device_reason(UnicodeDecodeError) is not None


class Shared(KeyError):
    pass


class Derived(Shared, RuntimeError):
    pass


def test_a_table_round_trips_its_qualified_names_with_the_same_ids() -> None:
    table = ErrorClassTable()
    identifier = table.register(Derived)
    loaded = ErrorClassTable.loads(table.dumps())
    assert loaded.register(Derived) == identifier == table.register(Derived)
    assert loaded.cls_of(identifier) is Derived
    assert loaded.matches(identifier, KeyError)


def test_a_class_another_process_cannot_find_is_refused_by_name() -> None:
    def make() -> type[BaseException]:
        class Local(RuntimeError):
            pass

        return Local

    table = ErrorClassTable()
    table.register(make())
    with pytest.raises(ValueError, match="not importable by name"):
        table.dumps()


def test_loads_refuses_names_it_cannot_resolve_or_ids_it_cannot_reproduce() -> None:
    with pytest.raises(ValueError, match="cannot be found"):
        ErrorClassTable.loads(json.dumps([["no_such_module", "Missing"]]))
    names = json.loads(ErrorClassTable().dumps())
    names.append(["builtins", "ValueError"])
    with pytest.raises(ValueError, match="different id"):
        ErrorClassTable.loads(json.dumps(names))
