"""Host raising: rebuilt classes against Python's own constructors, notes, group shapes, and KernelError."""

from operator import attrgetter

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from krn._internals.records import Message, Scalar, Tag, envelope_dtype
from krn.exceptions import KernelError, decode_error, encode_error, raise_errors, rebuild
from krn.exceptions.records import ErrorEvent
from krn.exceptions.schemas import SCHEMAS
from krn.exceptions.sites import ArgTag, ErrorSiteTable, RaiseSite
from krn.exceptions.table import ErrorClassTable

KERNEL = "saxpy"
ARG_COUNT = 4
ATTRIBUTES = ("code", "errno", "strerror", "filename", "filename2", "value", "name", "obj")
TEXT = st.text(alphabet="abc _=", max_size=4)
VALUES: dict[str, st.SearchStrategy[Scalar]] = {
    "int": st.integers(-(2**63), 2**63 - 1),
    "float": st.floats(allow_nan=False, allow_infinity=False),
    "bool": st.booleans(),
}
ARG_KINDS = {"int": Tag.INT, "float": Tag.FLOAT64, "bool": Tag.BOOL}


class OutOfBudget(RuntimeError):
    def __init__(self, limit: int, used: float, strict: bool) -> None:
        super().__init__(limit, used, strict)


class Labelled(ValueError):
    def __init__(self, what: str, count: int) -> None:
        super().__init__(what, count)


class Pair(RuntimeError):
    def __init__(self, first: int, second: int) -> None:
        super().__init__(first, second)


class Stop(BaseException):
    pass


class Changing(ValueError):
    def __init__(self, first: int) -> None:
        super().__init__(first)


class Interrupting(RuntimeError):
    def __init__(self, limit: int) -> None:
        raise KeyboardInterrupt


class Launch:
    """The sites and device records of one launch, encoded the way a backend writes them."""

    def __init__(self) -> None:
        self.table = ErrorClassTable()
        self.sites = ErrorSiteTable()
        self.records: list[np.ndarray] = []

    def site(self, cls: type[BaseException], *tags: ArgTag, line: int = 12) -> RaiseSite:
        return self.sites.add(
            kernel=KERNEL, file="saxpy.py", line=line, class_id=self.table.register(cls), arg_tags=tags
        )

    def raise_at(self, site: RaiseSite, block: int, thread: int, *values: int | float | bool) -> None:
        self.records.append(encode_error(site, block=block, thread=thread, values=values, slots=ARG_COUNT))

    def array(self) -> np.ndarray:
        return np.concatenate([np.zeros(0, dtype=envelope_dtype(ARG_COUNT)), *self.records])

    def events(self) -> list[ErrorEvent]:
        return [decode_error(record, self.sites) for record in self.array()]


def formatted(template: str, *tags: Tag) -> Message:
    return Message.of(template, list(tags))


def test_formatted_message_is_the_builtins_one_argument() -> None:
    launch = Launch()
    launch.raise_at(launch.site(ValueError, formatted("x={}", Tag.INT)), 2, 7, 3)
    error = rebuild(launch.table, launch.events()[0])
    assert type(error) is ValueError and error.args == ("x=3",)
    assert error.__notes__ == ["raised in kernel saxpy at saxpy.py:12", "block 2, thread 7"]


def test_builtin_arguments_stay_what_python_would_give() -> None:
    launch = Launch()
    launch.raise_at(launch.site(KeyError, Tag.INT), 0, 0, 7)
    launch.raise_at(launch.site(SystemExit, Tag.INT, line=13), 0, 1, 7)
    launch.raise_at(launch.site(StopIteration, Tag.FLOAT64, line=14), 0, 2, 2.5)
    key, exit_, stop = (rebuild(launch.table, event) for event in launch.events())
    assert repr(key) == "KeyError(7)" and type(key.args[0]) is int
    assert isinstance(exit_, SystemExit) and exit_.code == 7
    assert isinstance(stop, StopIteration) and stop.value == 2.5


def test_os_error_selects_its_subclass_from_the_recorded_errno() -> None:
    launch = Launch()
    launch.raise_at(launch.site(OSError, Tag.INT, Message.constant("no such file")), 0, 0, 2)
    error = rebuild(launch.table, launch.events()[0])
    assert type(error) is FileNotFoundError
    assert error.errno == 2 and error.strerror == "no such file"


@pytest.mark.parametrize("cls", [KeyboardInterrupt, GeneratorExit])
def test_base_exceptions_keep_their_arguments(cls: type[BaseException]) -> None:
    launch = Launch()
    launch.raise_at(launch.site(cls, Tag.INT, Message.constant("stop")), 0, 0, 4)
    assert rebuild(launch.table, launch.events()[0]).args == (4, "stop")


@settings(max_examples=400, deadline=None)
@given(st.sampled_from(sorted(SCHEMAS, key=attrgetter("__qualname__"))), st.data())
def test_rebuilt_builtin_is_python_s_own_call(cls: type[BaseException], data: st.DataObject) -> None:
    schema = SCHEMAS[cls]
    largest = ARG_COUNT if schema.most is None else min(schema.most, ARG_COUNT)
    tags: list[ArgTag] = []
    values: list[Scalar] = []
    expected: list[Scalar | str] = []
    for kind in data.draw(st.lists(st.sampled_from([*VALUES, "text"]), max_size=largest)):
        if kind == "text":
            text = data.draw(TEXT)
            tags.append(Message.constant(text))
            expected.append(text)
        else:
            value = data.draw(VALUES[kind])
            tags.append(ARG_KINDS[kind])
            values.append(value)
            expected.append(value)
    launch = Launch()
    launch.raise_at(launch.site(cls, *tags), 0, 0, *values)
    error = rebuild(launch.table, launch.events()[0])
    python = cls(*expected)
    assert type(error) is type(python)
    assert error.args == python.args
    assert repr(error) == repr(python)
    assert [getattr(error, name, None) for name in ATTRIBUTES] == [getattr(python, name, None) for name in ATTRIBUTES]


def test_user_class_is_rebuilt_from_its_arguments_with_constants_in_place() -> None:
    launch = Launch()
    launch.raise_at(launch.site(OutOfBudget, Tag.INT, Tag.FLOAT64, Tag.BOOL), 0, 1, 3, 4.5, True)
    launch.raise_at(launch.site(Labelled, Message.constant("tag"), Tag.INT), 0, 2, 5)
    budget, labelled = (rebuild(launch.table, event) for event in launch.events())
    assert type(budget) is OutOfBudget and budget.args == (3, 4.5, True)
    assert type(labelled) is Labelled and labelled.args == ("tag", 5)


@pytest.mark.parametrize(
    "others,note", [(1, "1 other thread raised the same site"), (3, "3 other threads raised the same site")]
)
def test_other_threads_on_the_same_site_are_counted_in_a_note(others: int, note: str) -> None:
    launch = Launch()
    launch.raise_at(launch.site(KeyError, Message.constant("k")), 0, 0)
    assert rebuild(launch.table, launch.events()[0], others=others).__notes__[-1] == note


def test_class_that_cannot_be_imported_becomes_a_kernel_error_with_its_site_and_no_cause() -> None:
    class Local(RuntimeError):
        pass

    launch = Launch()
    launch.raise_at(launch.site(Local, Tag.INT), 4, 9, 4)
    error = rebuild(launch.table, launch.events()[0], others=1)
    assert isinstance(error, KernelError) and isinstance(error, RuntimeError)
    assert str(error).endswith("Local: 4")
    assert error.__notes__[:2] == ["raised in kernel saxpy at saxpy.py:12", "block 4, thread 9"]
    assert error.__cause__ is None
    assert error.kernel == KERNEL and error.site == ("saxpy.py", 12)
    assert error.class_name.endswith("Local") and (error.block, error.thread, error.count) == (4, 9, 1)


def test_constructor_mismatch_keeps_the_failure_as_the_cause() -> None:
    launch = Launch()
    launch.raise_at(launch.site(Pair, Tag.INT), 0, 0, 1)
    error = rebuild(launch.table, launch.events()[0])
    assert isinstance(error, KernelError) and "Pair: 1" in str(error)
    assert isinstance(error.__cause__, TypeError)


def test_a_constructor_that_fails_after_the_class_changed_is_a_kernel_error_with_its_cause() -> None:
    launch = Launch()
    launch.raise_at(launch.site(Changing, Tag.INT, line=21), 3, 4, 8)

    def broken(self: Changing, first: int) -> None:
        raise ValueError("the class no longer accepts this")

    Changing.__init__ = broken
    error = rebuild(launch.table, launch.events()[0], others=2)
    assert isinstance(error, KernelError)
    assert isinstance(error.__cause__, ValueError) and str(error.__cause__) == "the class no longer accepts this"
    assert (error.kernel, error.site, error.block, error.thread, error.count) == (KERNEL, ("saxpy.py", 21), 3, 4, 2)
    assert error.__notes__[-1] == "2 other threads raised the same site"


def test_base_exceptions_from_a_constructor_are_not_kernel_errors() -> None:
    launch = Launch()
    launch.raise_at(launch.site(Interrupting, Tag.INT), 0, 0, 1)
    with pytest.raises(KeyboardInterrupt):
        rebuild(launch.table, launch.events()[0])


def test_empty_launch_raises_nothing() -> None:
    launch = Launch()
    raise_errors(launch.table, launch.sites, launch.array(), 0, KERNEL)


def test_one_error_is_raised_as_its_class_and_dropped_errors_are_noted() -> None:
    launch = Launch()
    launch.raise_at(launch.site(ValueError, formatted("x={}", Tag.INT)), 0, 3, 1)
    with pytest.raises(ValueError, match="x=1") as caught:
        raise_errors(launch.table, launch.sites, launch.array(), 2, KERNEL)
    assert caught.value.__notes__[-1] == "2 more errors were not recorded (buffer capacity 1)"


def test_many_threads_on_one_site_raise_one_error_from_the_first_thread() -> None:
    launch = Launch()
    site = launch.site(ValueError, formatted("x={}", Tag.INT))
    launch.raise_at(site, 1, 0, 8)
    launch.raise_at(site, 0, 2, 8)
    launch.raise_at(site, 0, 1, 8)
    with pytest.raises(ValueError) as caught:
        raise_errors(launch.table, launch.sites, launch.array(), 0, KERNEL)
    assert caught.value.__notes__ == [
        "raised in kernel saxpy at saxpy.py:12",
        "block 0, thread 1",
        "2 other threads raised the same site",
    ]


def test_many_sites_raise_a_group_named_after_the_kernel_in_first_thread_order() -> None:
    launch = Launch()
    value = launch.site(ValueError, formatted("x={}", Tag.INT), line=20)
    key = launch.site(KeyError, formatted("k={}", Tag.INT), line=30)
    launch.raise_at(value, 1, 0, 1)
    launch.raise_at(key, 0, 5, 2)
    launch.raise_at(value, 0, 2, 3)
    with pytest.raises(ExceptionGroup) as caught:
        raise_errors(launch.table, launch.sites, launch.array(), 1, KERNEL)
    group = caught.value
    assert group.message == KERNEL
    assert [type(error) for error in group.exceptions] == [ValueError, KeyError]
    assert group.exceptions[0].__notes__[-1] == "1 other thread raised the same site"
    assert group.__notes__ == [
        "3 threads raised 2 distinct errors",
        "1 more errors were not recorded (buffer capacity 3)",
    ]


def test_except_star_catches_each_class_of_the_group() -> None:
    launch = Launch()
    launch.raise_at(launch.site(ValueError, formatted("v"), line=20), 0, 1)
    launch.raise_at(launch.site(KeyError, formatted("k"), line=30), 0, 2)
    caught: list[tuple[str, int]] = []
    try:
        raise_errors(launch.table, launch.sites, launch.array(), 0, KERNEL)
    except* ValueError as group:
        caught.append(("value", len(group.exceptions)))
    except* KeyError as group:
        caught.append(("key", len(group.exceptions)))
    assert caught == [("value", 1), ("key", 1)]


def test_python_base_exception_classes_are_grouped_as_a_base_group() -> None:
    launch = Launch()
    launch.raise_at(launch.site(Stop), 0, 0)
    launch.raise_at(launch.site(ValueError, formatted("v"), line=20), 0, 1)
    with pytest.raises(BaseExceptionGroup) as caught:
        raise_errors(launch.table, launch.sites, launch.array(), 0, KERNEL)
    assert not isinstance(caught.value, ExceptionGroup)
    assert [type(error) for error in caught.value.exceptions] == [Stop, ValueError]
