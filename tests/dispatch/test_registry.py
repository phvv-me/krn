"""The overload registry: specificity, ties at registration, feature gating, refusals and the lookup error."""

import dataclasses

import pytest

from krn import dispatch
from krn.compiler.dtypes import F32, I8X4, U8X4, U32, sint
from krn.compiler.errors import Span, TargetError
from krn.compiler.ir import Const
from krn.dispatch import Expansion, Family, Instruction, Refusal, register, resolve
from krn.facts import catalog
from krn.facts.targets import Feature, Target

SPAN = Span("registry.py", 1, 1)


def constant(_operands, span: Span) -> Const:
    return Const(U32, 0, span=span)


def test_exact_dtype_beats_family_and_packed_beats_integer() -> None:
    assert isinstance(resolve("maximum", I8X4, Target.of("sm_90a")).lowering, Instruction)
    assert isinstance(resolve("maximum", sint(32), Target.of("sm_90a")).lowering, Expansion)
    assert isinstance(resolve("maximum", F32, Target.of("sm_90a")).lowering, Expansion)


def test_two_overloads_that_could_win_the_same_call_tie_at_registration() -> None:
    register("test.tie", Family.INT, Expansion(constant))
    with pytest.raises(ValueError, match="tie"):
        register("test.tie", Family.SIGNED, Expansion(constant))
    register("test.tie", Family.FLOAT, Expansion(constant))  # disjoint families do not tie


def test_a_lookup_with_no_candidate_names_them_and_the_target() -> None:
    with pytest.raises(TargetError) as error:
        resolve("add_sat", F32, Target.of("sm_90a"), SPAN)
    assert error.value.feature == "add_sat"
    assert error.value.target == "sm_90a"
    assert "no overload of add_sat for f32 on sm_90a" in error.value.reason
    assert "candidates: i8x4, u8x4, i16x2, u16x2" in error.value.reason
    assert error.value.instead and error.value.span == SPAN


def test_an_overload_needing_a_feature_applies_only_where_the_target_has_it() -> None:
    register("test.feature", sint(16), Expansion(constant))
    register("test.feature", sint(16), Expansion(constant), frozenset({Feature.TENSOR_CORE}))
    assert resolve("test.feature", sint(16), Target.of("spr")).features == {Feature.TENSOR_CORE}
    assert resolve("test.feature", sint(16), Target.of("avx512")).features == frozenset()


def test_a_missing_feature_with_no_fallback_is_a_lookup_error() -> None:
    register("test.gated", sint(8), Expansion(constant), frozenset({Feature.GRID_SYNC}))
    with pytest.raises(TargetError, match="candidates: i8"):
        resolve("test.gated", sint(8), Target.of("avx512"))


def test_refusal_names_its_reason_and_the_alternative() -> None:
    overload = dispatch.registry.Overload("test.refused", Family.ANY, Refusal("no unit", "use another target"))
    with pytest.raises(TargetError) as error:
        dispatch.lower(overload, (), SPAN, Target.of("avx512"))
    assert error.value.reason == "no unit" and error.value.instead == "use another target"


def test_an_instruction_without_a_spelling_on_the_target_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = dataclasses.replace(catalog.BY_NAME["simd.vaddss4"], nvidia="")
    monkeypatch.setitem(catalog.BY_NAME, "test.silent", entry)
    overload = dispatch.registry.Overload("test.silent", U8X4, Instruction("test.silent", frozenset({"nvidia"})))
    with pytest.raises(TargetError, match="has no lowering on sm_90a"):
        dispatch.lower(overload, (), SPAN, Target.of("sm_90a"))


def test_native_on_nvidia_is_one_instruction_and_a_composed_one_counts_its_statements() -> None:
    overload = resolve("add_sat", U8X4, Target.of("sm_120a"))
    native = dispatch.lower(overload, (_lane(), _lane()), SPAN, Target.of("sm_120a"))
    assert native.native and native.instructions == 1
    composed = dispatch.lower(overload, (_lane(), _lane()), SPAN, Target.of("gfx942"))
    assert not composed.native
    assert composed.instructions == catalog.BY_NAME["general.add_sat.u8x4"].helper_for("amd").count(";")


def test_an_expansion_counts_the_operations_it_emits() -> None:
    overload = resolve("add_sat", U32, Target.of("sm_120a"))
    built = dispatch.lower(overload, (_lane(U32), _lane(U32)), SPAN, Target.of("sm_120a"))
    assert not built.native and built.instructions == 4  # subtract, compare, add, select


def test_fallback_message_names_the_targets_where_it_is_native() -> None:
    overload = resolve("add_sat", U8X4, Target.of("gfx942"))
    built = dispatch.lower(overload, (_lane(), _lane()), SPAN, Target.of("gfx942"))
    message = dispatch.fallback_message(overload, built, U8X4, Target.of("gfx942"))
    assert message.startswith("add_sat on u8x4 has no native instruction on gfx942; it is emitted as a ")
    assert message.endswith("-instruction sequence; native on Apple GPUs and NVIDIA GPUs")
    plain = resolve("add_sat", U32, Target.of("sm_120a"))
    text = dispatch.fallback_message(
        plain, dispatch.lower(plain, (_lane(U32), _lane(U32)), SPAN, Target.of("sm_120a")), U32, Target.of("sm_120a")
    )
    assert text.endswith("not native on any GPU target")


def _lane(dtype=U8X4):
    from krn.compiler.ir import Var

    return Var(dtype, "x", span=SPAN)
