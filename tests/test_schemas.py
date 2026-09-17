"""Schema tests."""
import pytest
from pydantic import ValidationError

from overnight_runner.schemas import (
    Disposition,
    ExecutionClass,
    TaskManifest,
    canonical_sha,
)


def _base(**kw):
    base = dict(
        task_id="test-001",
        title="t",
        execution_class="read_only",
        objective="o",
        repo={"path": "/tmp/x"},
    )
    base.update(kw)
    return base


def test_minimal_manifest_ok():
    m = TaskManifest.model_validate(_base())
    assert m.task_id == "test-001"
    assert m.execution_class == ExecutionClass.READ_ONLY


def test_unknown_field_rejected():
    bad = _base()
    bad["extra_top_level"] = "nope"
    with pytest.raises(ValidationError):
        TaskManifest.model_validate(bad)


def test_read_only_cannot_declare_writes():
    with pytest.raises(ValidationError):
        TaskManifest.model_validate(_base(paths={"write_paths": ["x.py"]}))


def test_source_mutation_ok_with_writes():
    m = TaskManifest.model_validate(_base(
        execution_class="source_mutation",
        paths={"write_paths": ["hello.py"], "read_paths": ["hello.py"]},
    ))
    assert "hello.py" in m.paths.write_paths


def test_canonical_sha_deterministic():
    m1 = TaskManifest.model_validate(_base())
    m2 = TaskManifest.model_validate(_base())
    assert canonical_sha(m1) == canonical_sha(m2)


def test_wildcards_rejected():
    with pytest.raises(ValidationError):
        TaskManifest.model_validate(_base(
            execution_class="source_mutation",
            paths={"write_paths": ["*.py"]},
        ))


def test_path_traversal_rejected():
    with pytest.raises(ValidationError):
        TaskManifest.model_validate(_base(
            execution_class="source_mutation",
            paths={"write_paths": ["../escape.py"]},
        ))


def test_disposition_values():
    assert Disposition("DONE").value == "DONE"
    assert Disposition("REVIEW_REQUIRED").value == "REVIEW_REQUIRED"
