"""
tests/test_phase13_exceptions.py
==================================
Phase 13 exception hierarchy tests.

Verifies that:
1. All engine exceptions are importable
2. All inherit from GISProcessingError
3. The hierarchy is correct (GISProcessingError inherits from Exception)
4. Each exception can be raised and caught individually
5. The catch-all GISProcessingError correctly intercepts all subtypes
"""

import pytest
from src.gis_engine.core.exceptions import (
    GISProcessingError,
    InputValidationError,
    ConfigurationError,
    AlignmentError,
    InferenceError,
    HydrologyError,
    FusionError,
    MetricError,
    OutputError,
    ValidationError,
)

ALL_EXCEPTIONS = [
    InputValidationError,
    ConfigurationError,
    AlignmentError,
    InferenceError,
    HydrologyError,
    FusionError,
    MetricError,
    OutputError,
    ValidationError,
]


class TestExceptionHierarchy:
    """All engine exceptions must inherit from GISProcessingError."""

    def test_base_inherits_from_exception(self):
        assert issubclass(GISProcessingError, Exception)

    @pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
    def test_subclass_of_gis_processing_error(self, exc_cls):
        assert issubclass(exc_cls, GISProcessingError), (
            f"{exc_cls.__name__} does not inherit from GISProcessingError"
        )

    @pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
    def test_subclass_of_exception(self, exc_cls):
        assert issubclass(exc_cls, Exception)


class TestExceptionRaiseAndCatch:
    """Each exception can be raised and caught individually and by base class."""

    @pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
    def test_raise_and_catch_specific(self, exc_cls):
        with pytest.raises(exc_cls):
            raise exc_cls("test message")

    @pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
    def test_catch_as_base_class(self, exc_cls):
        """All subtypes must be caught with GISProcessingError."""
        with pytest.raises(GISProcessingError):
            raise exc_cls("caught as base class")

    @pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
    def test_message_preserved(self, exc_cls):
        msg = f"{exc_cls.__name__}: specific error message"
        exc = exc_cls(msg)
        assert str(exc) == msg

    @pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
    def test_exception_chaining(self, exc_cls):
        """Exceptions must support chaining (raise X from Y)."""
        original = ValueError("original cause")
        with pytest.raises(exc_cls) as exc_info:
            try:
                raise original
            except ValueError as e:
                raise exc_cls("wrapped error") from e
        assert exc_info.value.__cause__ is original


class TestExceptionDistinctness:
    """No two exception types should be the same class."""

    def test_all_distinct_classes(self):
        names = [cls.__name__ for cls in ALL_EXCEPTIONS]
        assert len(names) == len(set(names)), (
            "Duplicate exception class names detected!"
        )

    def test_all_distinct_types(self):
        for i, a in enumerate(ALL_EXCEPTIONS):
            for j, b in enumerate(ALL_EXCEPTIONS):
                if i != j:
                    assert a is not b, (
                        f"{a.__name__} and {b.__name__} are the same object"
                    )
