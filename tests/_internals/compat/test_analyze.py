"""The analyzer: each judge on the targets it decides, the four outcomes of a compile, and the probe builders."""

from dataclasses import replace

import pytest

from krn._internals.compat import Feature, Requires, Status, Support
from krn._internals.compat.analyze import (
    ALTERNATIVE,
    PACKED_TYPES,
    Probe,
    analyze,
    catalog_feature,
    catalog_judge,
    catalog_module,
    combine,
    conversion,
    dispatched,
    family,
    function_probes,
    matmul,
    spelled,
    type_probes,
)
from krn._internals.compat.requires import METAL_FEATURES, requirement
from krn.compiler.dtypes import F32, from_sketch
from krn.compiler.errors import TargetError
from krn.facts import catalog
from krn.facts.targets import Feature as TargetFeature
from krn.facts.targets import Target, all_targets
from krn.frontend import numerics

SM120A = Target.of("sm_120a")
SM80 = Target.of("sm_80")
APPLE = Target.of("apple_m4")
AVX512 = Target.of("avx512")
AVX2 = Target.of("avx2")
GFX942 = Target.of("gfx942")

NATIVE_SOURCE = """from krn import T, Vector, grid, kernel
@kernel
def probe(a: Vector[T.f32], out: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < out.shape[0]:
        out[i] = a[i] + a[i]
"""
UNDEFINED_SOURCE = """from krn import T, Vector, grid, kernel
@kernel
def probe(a: Vector[T.f32], out: Vector[T.f32]) -> None:
    i = grid.thread_rank()
    if i < out.shape[0]:
        out[i] = missing_name(a[i])
"""


def feature(identifier: str) -> Feature:
    return Feature(id=identifier, module="test", group="test", name=identifier)


def probe(identifier: str, judge, source: str | None = None) -> Probe:
    return Probe(feature=feature(identifier), judge=judge, source=source)


def native(target: Target) -> Support:
    return Support(status=Status.NATIVE)


def refuse_with_target_error(target: Target) -> Support:
    raise TargetError("test.op", target.arch, "the unit is absent", "use the other unit")


def test_compiled_native_probe_carries_what_the_target_needs() -> None:
    matrix = analyze([probe("add", native, NATIVE_SOURCE)], [SM120A, APPLE, AVX2], {})
    assert matrix.cells["add"]["sm_120a"] == Support(status=Status.NATIVE, requires=requirement(SM120A))
    assert matrix.cells["add"]["apple_m4"].requires == Requires(gpu="Apple9", source=METAL_FEATURES)
    assert matrix.cells["add"]["avx2"] == Support(status=Status.NATIVE, requires=requirement(AVX2))


def test_target_error_from_the_compile_is_refused_with_its_parts() -> None:
    source = """from krn import T, Vector, grid, kernel
from krn import intrinsics
@kernel
def probe(a: Vector[T.u32], out: Vector[T.u32]) -> None:
    i = grid.thread_rank()
    if i < out.shape[0]:
        out[i] = intrinsics.vaddss4(a[i], a[i])
"""
    matrix = analyze([probe("vaddss4", native, source)], [SM120A, APPLE], {})
    assert matrix.cells["vaddss4"]["sm_120a"].status is Status.NATIVE
    refused = matrix.cells["vaddss4"]["apple_m4"]
    assert refused.status is Status.REFUSED
    assert "NVIDIA SIMD instruction" in refused.note and refused.instead


def test_judge_raising_target_error_is_refused() -> None:
    matrix = analyze([probe("op", refuse_with_target_error)], [SM120A], {})
    assert matrix.cells["op"]["sm_120a"] == Support(
        status=Status.REFUSED, note="the unit is absent", instead="use the other unit"
    )


def test_other_capture_error_is_refused_with_the_generic_alternative() -> None:
    matrix = analyze([probe("bad", native, UNDEFINED_SOURCE)], [SM120A], {})
    refused = matrix.cells["bad"]["sm_120a"]
    assert refused.status is Status.REFUSED
    assert "missing_name" in refused.note and refused.instead == ALTERNATIVE


@pytest.mark.parametrize("error", [NotImplementedError("not lowered here"), ValueError("bad value")])
def test_unclassified_error_is_unknown_with_its_message(error: Exception) -> None:
    def judge(target: Target) -> Support:
        raise error

    matrix = analyze([probe("odd", judge)], [SM120A], {})
    assert matrix.cells["odd"]["sm_120a"] == Support(status=Status.UNKNOWN, note=str(error))


def test_composed_judge_keeps_its_length_and_probe_without_source_is_not_compiled() -> None:
    def composed(target: Target) -> Support:
        return Support(status=Status.COMPOSED, note="a sequence", length=5)

    matrix = analyze([probe("sequence", composed)], [AVX2], {"from": "test"})
    assert matrix.cells["sequence"]["avx2"] == Support(status=Status.COMPOSED, note="a sequence", length=5)
    assert matrix.generated_from == {"from": "test"}
    assert matrix.features[0].id == "sequence"


def test_repeated_feature_id_is_an_error() -> None:
    with pytest.raises(ValueError, match="listed twice"):
        analyze([probe("same", native), probe("same", native)], [SM120A], {})


def test_combine_reports_the_first_refusal_else_native_else_the_composed_sum() -> None:
    refused = Support(status=Status.REFUSED, note="no", instead="use f16")
    assert combine([Support(status=Status.NATIVE), refused]) == refused
    assert combine([Support(status=Status.UNKNOWN, note="x"), refused]).status is Status.UNKNOWN
    assert combine([]) == Support(status=Status.NATIVE)
    total = combine(
        [Support(status=Status.COMPOSED, note="a", length=2), Support(status=Status.COMPOSED, note="a", length=3)]
    )
    assert total == Support(status=Status.COMPOSED, note="a", length=5)
    assert combine([Support(status=Status.COMPOSED, note="b")]).length is None


def test_spelled_reports_refusals_and_helpers() -> None:
    entry = catalog.BY_NAME["bits.bfe"]
    assert spelled(replace(entry, helper="", nvidia="spelled"), SM120A) == Support(status=Status.NATIVE)
    helper = spelled(replace(entry, helper="int f() { return 1; }; int g();"), SM120A)
    assert helper.status is Status.COMPOSED and helper.length == 3
    assert "needs sm_100" in spelled(replace(entry, sm=100), SM80).note
    assert "needs tensor_memory" in spelled(replace(entry, feature=TargetFeature.TENSOR_MEMORY), SM120A).note
    assert "has no lowering" in spelled(replace(entry, nvidia=""), SM120A).note


def test_catalog_judge_uses_the_dispatcher_for_general_rows_and_all_parts_for_tuples() -> None:
    general = catalog.BY_NAME["general.add_sat.i8x4"]
    assert catalog_judge(general)(SM120A).status is Status.NATIVE
    assert catalog_judge(general)(AVX512).status is Status.COMPOSED
    sincos = catalog_judge(catalog.BY_NAME["libm.sincos"])
    assert sincos(AVX512).status in (Status.NATIVE, Status.COMPOSED)


def test_dispatched_judge_matches_the_dispatcher_lowering() -> None:
    assert dispatched("add_sat", from_sketch(numerics.i8x4))(SM120A).status is Status.NATIVE
    assert (dispatched("avg", from_sketch(numerics.i8x4))(AVX512).length or 0) > 1
    assert dispatched("maximum", F32)(AVX512) == Support(status=Status.NATIVE)
    with pytest.raises(TargetError, match="no overload of add_sat"):
        dispatched("add_sat", F32)(APPLE)


def test_conversion_is_composed_only_where_the_narrow_instruction_is_absent() -> None:
    fp8 = from_sketch(numerics.f8e4m3fn)
    assert conversion(fp8)(SM80).status is Status.COMPOSED
    assert conversion(fp8)(SM120A) == Support(status=Status.NATIVE)
    assert conversion(F32)(SM80) == Support(status=Status.NATIVE)


def test_matmul_is_native_on_a_matrix_unit_and_composed_on_the_vector_units() -> None:
    f16 = from_sketch(numerics.f16)
    assert matmul(f16)(SM120A) == Support(status=Status.NATIVE)
    assert matmul(f16)(AVX512).status is Status.COMPOSED


def test_the_compiler_refuses_narrow_arithmetic_and_f64_on_apple_and_passes_f16() -> None:
    probes = {p.feature.id: p for p in type_probes()}
    ids = ("f8e4m3fn.arithmetic", "f16.arithmetic", "f64.convert")
    cells = analyze([probes[identifier] for identifier in ids], [SM120A, APPLE], {}).cells
    assert cells["f8e4m3fn.arithmetic"][SM120A.arch].status is Status.REFUSED
    assert cells["f16.arithmetic"][SM120A.arch].status is Status.NATIVE
    assert cells["f64.convert"][APPLE.arch] == Support(
        status=Status.REFUSED,
        note="Apple GPUs have no 64-bit floating point",
        instead="use f32, or run on a CPU or NVIDIA target",
    )


def test_catalog_row_names_its_module_and_operand_types() -> None:
    assert catalog_module("fast.sin") == "krn.fast.math"
    assert catalog_module("general.add_sat.i8x4") == "krn.math"
    assert catalog_module("libm.erf") == "catalog"
    row = catalog_feature(catalog.BY_NAME["libm.erf"], ("f32",))
    assert row.id == "libm.erf[f32]" and row.signature.startswith("(") and row.doc
    assert catalog_feature(catalog.BY_NAME["general.add_sat.i8x4"]).signature == ""


def test_family_names_each_vendor() -> None:
    assert family(SM120A) == "sm_120" and family(GFX942) == "CDNA3" and family(APPLE) == "Apple family 9"
    assert family(AVX2) == "x86-64"
    assert {family(t) for t in all_targets()} >= {"CDNA4", "sm_80", "x86-64"}


def test_packed_words_have_lane_rows_and_refuse_a_float_conversion() -> None:
    probes = {p.feature.id: p for p in type_probes()}
    cells = analyze([probes[f"i8x4.{op}"] for op in ("arithmetic", "lanes", "convert")], [SM120A, AVX512], {}).cells
    assert cells["i8x4.arithmetic"][SM120A.arch] == Support(status=Status.NATIVE, requires=requirement(SM120A))
    assert cells["i8x4.lanes"][AVX512.arch].status is Status.NATIVE
    refused = cells["i8x4.convert"][AVX512.arch]
    assert refused.status is Status.REFUSED and "packed word" in refused.note and refused.instead


def test_type_probes_cover_every_number_type_and_refuse_fnuz_conversions() -> None:
    probes = {p.feature.id: p for p in type_probes()}
    assert len(probes) == 19 * 4 + 4 * len(PACKED_TYPES)
    assert probes["f32.matmul"].feature.module == "types"
    assert probes["f8e4m3fnuz.load_store"].judge(APPLE).status is Status.NATIVE
    cell = analyze([probes["f8e4m3fnuz.convert"]], [APPLE], {}).cells["f8e4m3fnuz.convert"][APPLE.arch]
    assert cell.status is Status.REFUSED and "FNUZ convention" in cell.note


def source_of(probes: dict[str, Probe], identifier: str) -> str:
    text = probes[identifier].source
    assert text is not None, identifier
    return text


def test_function_probes_call_each_function_on_its_operand_types() -> None:
    probes = {p.feature.id: p for p in function_probes()}
    for identifier in (
        "krn.math.sin[f32]",
        "krn.math.add_sat[u8]",
        "krn.math.ldexp[f32]",
        "krn.math.isnan[f64]",
        "krn.math.pack[u8]",
        "krn.math.pack[i16]",
        "krn.math.unpack[u8x4]",
        "krn.intrinsics.vaddss4[u32]",
    ):
        assert identifier in probes, identifier
    assert "ldexp(a0[i], a1[i])" in source_of(probes, "krn.math.ldexp[f32]")
    assert "T.u8(" in source_of(probes, "krn.math.isnan[f64]")
    assert "pack(a0[i], a1[i], a2[i], a3[i])" in source_of(probes, "krn.math.pack[u8]")
    assert "pack(a0[i], a1[i])" in source_of(probes, "krn.math.pack[i16]")
    assert "w0, w1, w2, w3 = " in source_of(probes, "krn.math.unpack[u8x4]")
    assert probes["krn.math.add_sat[u8]"].judge(SM120A).status is Status.COMPOSED
    assert probes["krn.math.add_sat[u8x4]"].judge(SM120A).status is Status.NATIVE
    assert probes["krn.math.sin[f32]"].judge(APPLE).status is Status.NATIVE
