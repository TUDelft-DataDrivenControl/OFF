"""Tests for OFF's automatic module compatibility checking."""

from __future__ import annotations

from itertools import permutations

import pytest

from off.OFFModule import CompatibilityLevel, OFFModule, check_compatibility, compatibility


pytestmark = pytest.mark.compatibility

# -------------------------------------------------
# Simple tests for the decorators and their inheritance
# -------------------------------------------------


class ExampleModule(OFFModule):
    """Small concrete module used to test compatibility discovery."""

    MODULE_TYPE = "ExampleModule"

    def step(self, it: int) -> None:
        pass

    def reset(self) -> None:
        pass

    @compatibility(CompatibilityLevel.FULL)
    def obs_full(self) -> None:
        pass

    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_optional(self) -> None:
        pass
    
    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_optional_inherit_no_decorator(self) -> None:
        pass

    @compatibility(CompatibilityLevel.FULL)   
    def obs_overwite_full_to_optional(self) -> None:
        pass

    @compatibility(CompatibilityLevel.OPTIONAL)   
    def obs_overwite_optional_to_full(self) -> None:
        pass

    def obs_undecorated(self) -> None:
        pass

    @compatibility(CompatibilityLevel.FULL)
    def helper(self) -> None:
        """A decorated non-observable must not be advertised."""


class ExampleChild(ExampleModule):

    @compatibility(CompatibilityLevel.FULL)
    def obs_child(self) -> None:
        pass

    def obs_optional_inherit_no_decorator(self) -> None:
        pass
    
    @compatibility(CompatibilityLevel.OPTIONAL)   
    def obs_overwite_full_to_optional(self) -> None:
        pass

    @compatibility(CompatibilityLevel.FULL)   
    def obs_overwite_optional_to_full(self) -> None:
        pass


@pytest.mark.compatibility
def test_decorated_observables_are_discovered() -> None:
    assert ExampleModule.compatibility.provides["obs_full"] is CompatibilityLevel.FULL
    assert ExampleModule.compatibility.provides["obs_optional"] is CompatibilityLevel.OPTIONAL

@pytest.mark.compatibility
def test_undecorated_observable_defaults_to_unknown() -> None:
    assert ExampleModule.compatibility.provides["obs_undecorated"] is CompatibilityLevel.UNKNOWN

@pytest.mark.compatibility   
def test_non_observable_method_is_not_advertised() -> None:
    assert "helper" not in ExampleModule.compatibility.provides

@pytest.mark.compatibility
def test_module_type_is_recorded() -> None:
    assert ExampleModule.compatibility.module_type == "ExampleModule"

@pytest.mark.compatibility
def test_module_type_is_inherited() -> None:
    assert ExampleChild.compatibility.module_type == "ExampleModule"

@pytest.mark.compatibility
def test_subclass_inherits_observables_and_adds_its_own() -> None:
    assert ExampleChild.compatibility.provides["obs_full"] is CompatibilityLevel.FULL
    assert ExampleChild.compatibility.provides["obs_optional"] is CompatibilityLevel.OPTIONAL
    assert ExampleChild.compatibility.provides["obs_child"] is CompatibilityLevel.FULL

@pytest.mark.compatibility
def test_subclass_does_not_inherit_compatibility() -> None:
    assert ExampleChild.compatibility.provides["obs_optional_inherit_no_decorator"] is CompatibilityLevel.UNKNOWN

@pytest.mark.compatibility
def test_subclass_overwrites_compatibility() -> None:
    assert ExampleModule.compatibility.provides["obs_overwite_full_to_optional"] is CompatibilityLevel.FULL
    assert ExampleModule.compatibility.provides["obs_overwite_optional_to_full"] is CompatibilityLevel.OPTIONAL
    assert ExampleChild.compatibility.provides["obs_overwite_full_to_optional"] is CompatibilityLevel.OPTIONAL
    assert ExampleChild.compatibility.provides["obs_overwite_optional_to_full"] is CompatibilityLevel.FULL


# --------------------------------------------------------
# Simple tests for the requirements and their inheritance
# --------------------------------------------------------
class AtmosphericModel(OFFModule):
    MODULE_TYPE = "AtmosphericModel"

    REQUIRES = {
        'ExampleModule' : {
            'obs_optional':            CompatibilityLevel.OPTIONAL,
            'obs_full':                CompatibilityLevel.FULL,
            'obs_overwite_full_to_optional':    CompatibilityLevel.FULL,
            'obs_overwite_optional_to_full':    CompatibilityLevel.OPTIONAL
        }, 
        'SecondModel' : {
            'obs_1':                   CompatibilityLevel.OPTIONAL,
            'obs_2':                   CompatibilityLevel.FULL
        }
    }

class AtmosphericModel_V2(AtmosphericModel):
    REQUIRES = {
        'ExampleModule': {
            'obs_overwite_full_to_optional':    CompatibilityLevel.OPTIONAL,
            'obs_overwite_optional_to_full':    CompatibilityLevel.FULL
        }
    }

@pytest.mark.compatibility
def test_requirements_correct_initialization() -> None:
    assert AtmosphericModel.compatibility.requires == {
        'ExampleModule' : {
            'obs_optional':            CompatibilityLevel.OPTIONAL,
            'obs_full':                CompatibilityLevel.FULL,
            'obs_overwite_full_to_optional':    CompatibilityLevel.FULL,
            'obs_overwite_optional_to_full':    CompatibilityLevel.OPTIONAL
        }, 
        'SecondModel' : {
            'obs_1':                   CompatibilityLevel.OPTIONAL,
            'obs_2':                   CompatibilityLevel.FULL
        }
    }

@pytest.mark.compatibility
def test_requirements_overwritten_by_subclass() -> None:
    assert AtmosphericModel_V2.compatibility.requires == {
        'ExampleModule' : {
            'obs_optional':            CompatibilityLevel.OPTIONAL,
            'obs_full':                CompatibilityLevel.FULL,
            'obs_overwite_full_to_optional':    CompatibilityLevel.OPTIONAL,
            'obs_overwite_optional_to_full':    CompatibilityLevel.FULL
        }, 
        'SecondModel' : {
            'obs_1':                   CompatibilityLevel.OPTIONAL,
            'obs_2':                   CompatibilityLevel.FULL
        }
    }

# ---------------------------------------------
# Tests related to compatibility checking 
# ---------------------------------------------

class ExampleAtmosphericModelMatcher(OFFModule):
    MODULE_TYPE = "SecondModel"

    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_1(self) -> None:
        pass

    @compatibility(CompatibilityLevel.FULL)
    def obs_2(self) -> None:
        pass

class ExampleAtmosphericModelMismatcher(OFFModule):
    MODULE_TYPE = "SecondModel"

    @compatibility(CompatibilityLevel.FULL)
    def obs_1(self) -> None:
        pass

    @compatibility(CompatibilityLevel.OPTIONAL)
    def obs_2(self) -> None:
        pass

@pytest.mark.compatibility
def test_compat_report_missing_components() -> None:
    ret = check_compatibility([ExampleModule, AtmosphericModel]) 
    assert set(ret) == {
        ('AtmosphericModel', 'SecondModel', 'obs_1', CompatibilityLevel.OPTIONAL, CompatibilityLevel.NONE), 
        ('AtmosphericModel', 'SecondModel', 'obs_2', CompatibilityLevel.FULL, CompatibilityLevel.NONE)
    }

@pytest.mark.compatibility
def test_compat_overwrite_missing_components() -> None:
    ret = check_compatibility([ExampleChild, AtmosphericModel]) 
    assert set(ret) == {
        ('AtmosphericModel', 'SecondModel', 'obs_1', CompatibilityLevel.OPTIONAL, CompatibilityLevel.NONE), 
        ('AtmosphericModel', 'SecondModel', 'obs_2', CompatibilityLevel.FULL, CompatibilityLevel.NONE),
        ('AtmosphericModel', 'ExampleModule', 'obs_overwite_full_to_optional', CompatibilityLevel.FULL, CompatibilityLevel.OPTIONAL),
    }

@pytest.mark.compatibility
def test_compat_overwrite_missing_components_V2() -> None:
    ret = check_compatibility([ExampleChild, AtmosphericModel_V2]) 
    assert set(ret) == {
        ('AtmosphericModel', 'SecondModel', 'obs_1', CompatibilityLevel.OPTIONAL, CompatibilityLevel.NONE), 
        ('AtmosphericModel', 'SecondModel', 'obs_2', CompatibilityLevel.FULL, CompatibilityLevel.NONE),
    }


@pytest.mark.compatibility
def test_all_modules_of_required_type_must_meet_requirement() -> None:
    class ExampleConsumer(OFFModule):
        MODULE_TYPE = "ExampleConsumer"
        REQUIRES = {
            "ExampleModule": {
                "obs_overwite_full_to_optional": CompatibilityLevel.FULL,
            }
        }

    ret = check_compatibility([ExampleConsumer, ExampleModule, ExampleChild])

    assert ret == [
        (
            "ExampleConsumer",
            "ExampleModule",
            "obs_overwite_full_to_optional",
            CompatibilityLevel.FULL,
            CompatibilityLevel.OPTIONAL,
        )
    ]


@pytest.mark.compatibility
def test_missing_observable_on_one_module_counts_as_none() -> None:
    class ExampleConsumer(OFFModule):
        MODULE_TYPE = "ExampleConsumer"
        REQUIRES = {"ExampleModule": {"obs_full": CompatibilityLevel.FULL}}

    class ExampleWithoutFullObservable(ExampleModule):
        obs_full = None

    ret = check_compatibility(
        [ExampleConsumer, ExampleModule, ExampleWithoutFullObservable]
    )

    assert ret == [
        (
            "ExampleConsumer",
            "ExampleModule",
            "obs_full",
            CompatibilityLevel.FULL,
            CompatibilityLevel.NONE,
        )
    ]


@pytest.mark.compatibility
def test_3_compatible_modules_return_non() -> None:
    print(ExampleAtmosphericModelMatcher.compatibility)
    print(AtmosphericModel.compatibility)
    ret = check_compatibility([AtmosphericModel, ExampleModule, ExampleAtmosphericModelMatcher])
    assert ret == []

@pytest.mark.compatibility
def test_3_mismatched_modules_missing_components() -> None:
    ret = check_compatibility([AtmosphericModel, ExampleModule, ExampleAtmosphericModelMismatcher])
    assert set(ret) == {
        ('AtmosphericModel', 'SecondModel', 'obs_2', CompatibilityLevel.FULL, CompatibilityLevel.OPTIONAL),
    }


@pytest.mark.compatibility
def test_module_order_does_not_change_compatibility_report() -> None:
    modules = [AtmosphericModel, ExampleChild, ExampleAtmosphericModelMismatcher]
    expected = set(check_compatibility(modules))

    for module_order in permutations(modules):
        assert set(check_compatibility(module_order)) == expected


@pytest.mark.compatibility
def test_check_compatibility_accepts_instances() -> None:
    assert check_compatibility([ExampleModule()]) == check_compatibility([ExampleModule])


@pytest.mark.compatibility
@pytest.mark.parametrize("invalid_module", [object(), object])
def test_check_compatibility_rejects_invalid_input(invalid_module: object) -> None:
    with pytest.raises(TypeError, match="Expected OFFModule class or instance"):
        check_compatibility([invalid_module])


@pytest.mark.compatibility
@pytest.mark.parametrize(("provided_level", "required_level", "is_unmet"), [
        (CompatibilityLevel.NONE, CompatibilityLevel.NONE, False),
        (CompatibilityLevel.NONE, CompatibilityLevel.UNKNOWN, True),
        (CompatibilityLevel.UNKNOWN, CompatibilityLevel.UNKNOWN, False),
        (CompatibilityLevel.UNKNOWN, CompatibilityLevel.OPTIONAL, True),
        (CompatibilityLevel.OPTIONAL, CompatibilityLevel.UNKNOWN, False),
        (CompatibilityLevel.OPTIONAL, CompatibilityLevel.FULL, True),
        (CompatibilityLevel.FULL, CompatibilityLevel.OPTIONAL, False),
        (CompatibilityLevel.FULL, CompatibilityLevel.FULL, False),
        ],
    )
def test_compatibility_level_thresholds(provided_level: CompatibilityLevel, required_level: CompatibilityLevel, is_unmet: bool) -> None:
    class ThresholdProvider(OFFModule):
        MODULE_TYPE = "ThresholdProvider"

        @compatibility(provided_level)
        def obs_value(self) -> None:
            pass

    class ThresholdConsumer(OFFModule):
        MODULE_TYPE = "ThresholdConsumer"
        REQUIRES = {"ThresholdProvider": {"obs_value": required_level}}

    report = check_compatibility([ThresholdProvider, ThresholdConsumer])

    assert bool(report) is is_unmet
    if is_unmet:
        assert report == [
            ("ThresholdConsumer", "ThresholdProvider", "obs_value", required_level, provided_level,)
        ]


@pytest.mark.compatibility
def test_empty_module_list_is_compatible() -> None:
    assert check_compatibility([]) == []


@pytest.mark.compatibility
def test_describe_compatibility_returns_report(capsys: pytest.CaptureFixture[str]) -> None:
    report = ExampleModule.describe_compatibility()

    captured = capsys.readouterr()
    assert report is not None
    assert captured.out == f"{report}\n"
    assert "Compatibility report for ExampleModule" in report
    assert "obs_full: full" in report
