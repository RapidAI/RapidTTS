# -*- encoding: utf-8 -*-
import ast
from pathlib import Path


def _load_requirement_groups():
    groups = {}
    current_group = None
    for raw_line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# [") and line.endswith("]"):
            current_group = line[3:-1].strip()
            groups[current_group] = []
            continue
        if line.startswith("#"):
            continue
        assert current_group is not None, f"ungrouped requirement: {line}"
        groups[current_group].append(line)
    return groups


def _load_setup_ast():
    return ast.parse(Path("setup.py").read_text(encoding="utf-8"))


def _assignment_value(setup_ast, name):
    for node in setup_ast.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                return node.value
    raise AssertionError(f"Missing setup.py assignment: {name}")


def _load_setup_call_keywords():
    setup_ast = ast.parse(Path("setup.py").read_text(encoding="utf-8"))
    for node in setup_ast.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr == "setup":
                return {keyword.arg: keyword.value for keyword in call.keywords}
    raise AssertionError("Missing setuptools.setup call")


def test_setup_default_dependencies_exclude_backend_specific_packages():
    groups = _load_requirement_groups()
    default_requires = groups["common"]

    assert "numpy>=1.22.3" in default_requires
    assert "soundfile>=0.12.1,<1.0" in default_requires
    assert "misaki-fork[zh]>=0.9.6,<1.0" not in default_requires
    assert "phonemizer-fork>=3.3.2,<4.0" not in default_requires
    assert "espeakng-loader>=0.2.4,<1.0" not in default_requires
    assert "g2p_en>=2.1.0,<3.0" not in default_requires
    assert "librosa>=0.11.0,<1.0" not in default_requires
    assert "sentencepiece>=0.2.0,<1.0" not in default_requires
    assert "tokenizers>=0.13.3,<1.0" not in default_requires


def test_setup_kokoro_extra_dependencies_are_separate():
    groups = _load_requirement_groups()
    kokoro_requires = groups["kokoro"]

    assert "misaki-fork[zh]>=0.9.6,<1.0" in kokoro_requires
    assert "phonemizer-fork>=3.3.2,<4.0" in kokoro_requires
    assert "espeakng-loader>=0.2.4,<1.0" in kokoro_requires


def test_setup_melo_extra_dependencies_are_separate():
    groups = _load_requirement_groups()
    melo_requires = groups["melo"]

    assert "g2p_en>=2.1.0,<3.0" in melo_requires
    assert "librosa>=0.11.0,<1.0" in melo_requires
    assert "tokenizers>=0.13.3,<1.0" in melo_requires


def test_setup_moss_nano_extra_dependencies_are_separate():
    groups = _load_requirement_groups()
    moss_nano_requires = groups["moss_nano"]

    assert "librosa>=0.11.0,<1.0" in moss_nano_requires
    assert "sentencepiece>=0.2.0,<1.0" in moss_nano_requires


def test_setup_reads_dependency_groups_from_requirements_file():
    setup_ast = _load_setup_ast()
    requirement_groups = _assignment_value(setup_ast, "REQUIREMENT_GROUPS")
    assert isinstance(requirement_groups, ast.Call)
    assert isinstance(requirement_groups.func, ast.Name)
    assert requirement_groups.func.id == "read_requirement_groups"
    assert ast.literal_eval(requirement_groups.args[0]) == "requirements.txt"

    common_requires = _assignment_value(setup_ast, "COMMON_REQUIRES")
    kokoro_requires = _assignment_value(setup_ast, "KOKORO_REQUIRES")
    melo_requires = _assignment_value(setup_ast, "MELO_REQUIRES")
    moss_nano_requires = _assignment_value(setup_ast, "MOSS_NANO_REQUIRES")

    assert common_requires.slice.value == "common"
    assert kokoro_requires.slice.value == "kokoro"
    assert melo_requires.slice.value == "melo"
    assert moss_nano_requires.slice.value == "moss_nano"


def test_setup_install_requires_and_extras_use_group_variables():
    keywords = _load_setup_call_keywords()

    install_requires = keywords["install_requires"]
    assert isinstance(install_requires, ast.Name)
    assert install_requires.id == "COMMON_REQUIRES"

    extras = keywords["extras_require"]
    assert isinstance(extras, ast.Dict)
    extra_keys = [ast.literal_eval(key) for key in extras.keys]
    assert extra_keys == ["kokoro", "melo", "moss_nano", "all"]
