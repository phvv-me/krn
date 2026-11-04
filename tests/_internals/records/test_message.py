"""Messages parse from f-strings, t-strings and `{}` text into typed fields, render in the printf subset, or refuse."""

import ast
import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from krn._internals.records import Argument, Field, Message, MessageFormatError, Spec, Tag
from krn._internals.records.jsonio import json_list, json_object
from tests._internals.records.strategies import spec_cases, values_for


def fstring(source: str) -> ast.JoinedStr:
    node = ast.parse(source, mode="eval").body
    assert isinstance(node, ast.JoinedStr)
    return node


def tstring(source: str, names: dict[str, int]):
    return eval(source, {}, names)


def test_fstring_fields_and_spec() -> None:
    message = Message.of(fstring('f"i={i} a={a:+.3f}"'), [Tag.INT, Tag.FLOAT64])
    assert message.format == "i={} a={}"
    assert [field.tag for field in message.fields] == [Tag.INT, Tag.FLOAT64]
    assert message.render([3, 1.5]) == "i=3 a=+1.500"
    assert Message.of(fstring('f"plain"')) == Message(format="plain", fields=())


def test_tstring_alignment_and_width() -> None:
    message = Message.of(tstring('t"x[{i}]={x:>4d}"', {"i": 1, "x": 2}), [Tag.INT, Tag.UINT])
    assert message.format == "x[{}]={}"
    assert message.render([1, 2]) == "x[1]=   2"


def test_braces_numbering_constants_and_escapes() -> None:
    message = Message.of("{{x}} {0:>5d} {1} {0:d}", [Tag.INT, "saxpy"])
    assert message.format == "{{x}} {} {} {}"
    assert [field.tag for field in message.fields] == [Tag.INT, Tag.STR_CONSTANT, Tag.INT]
    assert message.scalars == (message.fields[0], message.fields[2])
    assert message.render([7, 7]) == "{x}     7 saxpy 7"
    assert Message.of("{:d}", [Tag.INT]).render([5]) == "5"


def test_constant_text_is_captured_and_escaped_not_sent() -> None:
    message = Message.of("{} and {}", ["a {literal}", Tag.BOOL])
    assert message.fields[0] == Field(position=0, tag=Tag.STR_CONSTANT, text="a {literal}")
    assert message.render([True]) == "a {literal} and 1"
    assert Message.of("{{}}").render([]) == "{}"


def test_bool_is_zero_or_one_and_char_is_ascii() -> None:
    assert Message.of("{:>6}|", [Tag.BOOL]).render([False]) == "     0|"
    assert Message.of("{:c}", [Tag.UINT]).render([65]) == "A"
    assert Message.of("{:d}", [Tag.BOOL]).render([True]) == "1"


def test_negative_int_in_hex_is_its_sixty_four_bit_two_complement() -> None:
    assert Message.of("{:x}", [Tag.INT]).render([-1]) == "ffffffffffffffff"
    assert Message.of("{:o}", [Tag.INT]).render([-8]) == "1777777777777777777770"
    assert Message.of("{:u}", [Tag.INT]).render([-1]) == "18446744073709551615"
    assert Message.of("{:d}", [Tag.INT]).render([-1]) == "-1"


def test_non_finite_floats_pad_with_spaces_like_c_even_with_the_zero_flag() -> None:
    assert Message.of("{:08f}|", [Tag.FLOAT64]).render([float("inf")]) == "     inf|"
    assert Message.of("{:08f}|", [Tag.FLOAT64]).render([float("nan")]) == "     nan|"
    assert Message.of("{:08.2f}|", [Tag.FLOAT64]).render([-1.5]) == "-0001.50|"


def test_float_needs_a_printf_type_and_renders_its_letter() -> None:
    assert Message.of("{:.3e}", [Tag.FLOAT32]).render([12345.678]) == "1.235e+04"
    assert Message.of("{:G}", [Tag.FLOAT64]).render([0.0001]) == "0.0001"


@pytest.mark.parametrize(
    "source,args,rule",
    [
        ("{:^5}", [Tag.INT], "not supported"),
        ("{:x<5}", [Tag.INT], "not supported"),
        ("{:#x}", [Tag.UINT], "not supported"),
        ("{:,}", [Tag.INT], "not supported"),
        ("{:-}", [Tag.INT], "not supported"),
        ("{:{0}}", [Tag.INT], "not supported"),
        ("{!x}", [Tag.INT], "conversion !x has no printf form"),
        ("{!r}", [Tag.INT], "conversion !r has no printf form"),
        ("{!a:d}", [Tag.INT], "conversion !a has no printf form"),
        ("{!s}", ["saxpy"], "conversion !s has no printf form"),
        ("{:b}", [Tag.UINT], "binary b has no printf form"),
        ("{name}", [Tag.INT], "not supported"),
        ("{0.real}", [Tag.INT], "not supported"),
        ("{0[1]}", [Tag.INT], "not supported"),
        ("{", [Tag.INT], "malformed"),
        ("{} {0}", [Tag.INT, Tag.INT], "not both"),
        ("{}", [], "placeholder 0 has no argument"),
        ("{1}", [Tag.INT], "placeholder 1 has no argument"),
        ("x", [Tag.INT], "argument 0 has no placeholder"),
        ("{:f}", [Tag.BOOL], "type 'f' is not allowed for bool"),
        ("{:s}", [Tag.INT], "type 's' is not allowed for int"),
        ("{:d}", ["saxpy"], "type 'd' is not allowed for str-constant"),
        ("{}", [Tag.FLOAT64], "a float needs a printf type"),
        ("{:+x}", [Tag.UINT], "a sign is only for signed numbers"),
        ("{:+c}", [Tag.UINT], "a sign is only for signed numbers"),
        ("{:+}", ["saxpy"], "a sign is only for signed numbers"),
        ("{:05c}", [Tag.UINT], "0 padding"),
        ("{:<05}", [Tag.INT], "0 padding"),
        ("{:05}", ["saxpy"], "0 padding"),
        ("{:.2d}", [Tag.INT], "precision is only"),
        ("{:.2}", [Tag.BOOL], "precision is only"),
        ("{}", [Tag.STR_CONSTANT], "str argument"),
    ],
)
def test_unsupported_input_names_the_rule(source: str, args: list[Tag | str], rule: str) -> None:
    with pytest.raises(MessageFormatError, match=rule):
        Message.of(source, args)


def test_fstring_rejections() -> None:
    with pytest.raises(MessageFormatError, match="replacement fields"):
        Message.of(fstring('f"{x:{w}}"'), [Tag.INT])
    with pytest.raises(MessageFormatError, match="placeholder 1 has no argument"):
        Message.of(fstring('f"{a}{b}"'), [Tag.INT])
    with pytest.raises(MessageFormatError, match="argument 1 has no placeholder"):
        Message.of(fstring('f"{a}"'), [Tag.INT, Tag.INT])
    with pytest.raises(MessageFormatError, match="conversion !r"):
        Message.of(fstring('f"{a!r}"'), [Tag.INT])


def test_spec_names_its_parts_and_refuses_what_printf_cannot_say() -> None:
    spec = Spec.parse("+012.3e")
    assert (spec.align, spec.sign, spec.zero, spec.width, spec.precision, spec.type) == ("", "+", True, 12, 3, "e")
    assert Spec.parse("<2") == Spec(align="<", width=2)
    assert Spec.parse("") == Spec()
    with pytest.raises(MessageFormatError, match="not supported"):
        Spec.parse("%5d")


def test_spec_refuses_values_outside_the_alphabet() -> None:
    with pytest.raises(MessageFormatError, match="not in the printf subset"):
        Spec(align="^")
    with pytest.raises(MessageFormatError, match="negative"):
        Spec(width=-1)


def test_character_codes_outside_ascii_are_refused_when_rendered() -> None:
    message = Message.of("{:c}", [Tag.UINT])
    assert message.render([127]) == "\x7f"
    with pytest.raises(MessageFormatError, match="from 0 to 127, not 128"):
        message.render([128])
    with pytest.raises(MessageFormatError, match="from 0 to 127, not 18446744073709551615"):
        message.render([2**64 - 1])


def test_rendering_takes_one_value_per_scalar_field() -> None:
    with pytest.raises(ValueError, match="takes 1 values, not 2"):
        Message.of("{}", [Tag.INT]).render([1, 2])


@given(st.text(max_size=12), st.sampled_from(list(Tag)))
def test_any_text_parses_or_raises_the_format_error(text: str, tag: Tag) -> None:
    try:
        message = Message.of(text, [tag])
    except MessageFormatError:
        return
    assert len(message.fields) == 1


@given(spec_cases())
def test_every_case_the_contract_accepts_round_trips_through_json(case: tuple[Argument, str]) -> None:
    argument, text = case
    message = Message.of("x{:" + text + "}y", [argument])
    assert Message.from_dict(json.loads(json.dumps(message.to_dict()))) == message


@given(spec_cases(), st.data())
def test_accepted_cases_render_without_error(case: tuple[Argument, str], data: st.DataObject) -> None:
    argument, text = case
    message = Message.of("{:" + text + "}", [argument])
    field = message.fields[0]
    values = [] if field.tag is Tag.STR_CONSTANT else [data.draw(values_for(field.tag, field.letter))]
    assert isinstance(message.render(values), str)


def test_json_rejects_malformed_fields_by_name() -> None:
    item = Message.of("{:x}", [Tag.UINT]).to_dict()
    [field] = json_list(item, "fields")
    json_object(field)["tag"] = "int64"
    with pytest.raises(ValueError, match="not a valid"):
        Message.from_dict(item)


@given(st.text(max_size=12))
def test_a_constant_message_renders_its_text_whatever_its_braces(text: str) -> None:
    message = Message.constant(text)
    assert message.fields == () and message.scalars == ()
    assert message.render([]) == text
    assert Message.from_dict(json.loads(json.dumps(message.to_dict()))) == message
