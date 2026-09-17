"""Strict Pydantic schemas for task contracts.

All manifests MUST be validated and canonicalized before use.
Unknown fields are REJECTED (extra="forbid").
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated


SCHEMA_VERSION = "1.0"


# ----------------------------- Enums -----------------------------

class ExecutionClass(str, Enum):
    READ_ONLY = "read_only"
    SOURCE_MUTATION = "source_mutation"
    UNREAL_EDITOR = "unreal_editor"


class EditorAccess(str, Enum):
    NONE = "none"
    NON_EDITOR_ONLY = "non_editor_only"
    EDITOR_AUTOMATION = "editor_automation"


class Disposition(str, Enum):
    DONE = "DONE"
    BLOCKED = "BLOCKED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


class TaskStatus(str, Enum):
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


# ----------------------------- Helpers -----------------------------

_ID = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9._:-]{1,128}$")]


def canonical_json(data: dict[str, Any] | BaseModel) -> bytes:
    """Deterministic JSON: sort_keys, compact separators, UTF-8."""
    if isinstance(data, BaseModel):
        data = data.model_dump(mode="json", exclude_none=False)
    return json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json", exclude_none=False)
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_sha(model: BaseModel) -> str:
    return sha256_bytes(canonical_json(model))


# ----------------------------- Strict base -----------------------------

class StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False, str_strip_whitespace=False)


# ----------------------------- Pieces -----------------------------

class Constraints(StrictBase):
    no_unrelated_changes: bool = True
    prefer_smallest_change: bool = True


class PathPolicy(StrictBase):
    read_paths: list[str] = Field(default_factory=list)
    write_paths: list[str] = Field(default_factory=list)
    create_paths: list[str] = Field(default_factory=list)
    protected_read_paths: list[str] = Field(default_factory=list)


class CommandPolicy(StrictBase):
    model_allowed_command_ids: list[_ID] = Field(default_factory=list)
    required_validator_ids: list[_ID] = Field(default_factory=list)
    # If True, the manifest author accepts that zero files may change even on a
    # source_mutation task. Default False for v1 safety: source_mutation MUST
    # apply at least one change to PASS.
    allow_no_mutation: bool = False


class Limits(StrictBase):
    task_timeout_seconds: int = Field(default=900, ge=10, le=28800)
    max_model_turns: int = Field(default=8, ge=1, le=64)
    max_tool_calls: int = Field(default=20, ge=1, le=256)
    max_changed_files: int = Field(default=2, ge=0, le=16)
    max_diff_lines: int = Field(default=400, ge=1, le=20000)
    max_written_bytes: int = Field(default=65536, ge=1, le=4_000_000)
    max_tool_result_bytes: int = Field(default=24576, ge=256, le=1_000_000)


class ContextBudget(StrictBase):
    target_total_tokens: int = Field(default=8192, ge=512, le=32768)
    max_contract_tokens: int = Field(default=2048, ge=128, le=16384)
    max_read_bytes: int = Field(default=16384, ge=256, le=262144)
    max_files_read: int = Field(default=4, ge=0, le=64)
    max_files_written: int = Field(default=2, ge=0, le=16)


class ModelProfile(StrictBase):
    model_name: str = Field(default="gemma4:12b", min_length=1, max_length=200)
    num_ctx: int = Field(default=8192, ge=512, le=32768)
    num_predict: int = Field(default=1024, ge=64, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: int | None = Field(default=None)


class Repo(StrictBase):
    path: str = Field(min_length=1, max_length=4096)
    base_branch: str = Field(default="HEAD", min_length=1, max_length=200)
    require_clean_tree: bool = True


class Unreal(StrictBase):
    editor_access: EditorAccess = EditorAccess.NONE
    automation_test_names: list[str] = Field(default_factory=list)

    @field_validator("automation_test_names")
    @classmethod
    def _no_test_names_unless_editor(cls, v: list[str], info: Any) -> list[str]:
        ea = info.data.get("editor_access")
        if v and ea != EditorAccess.EDITOR_AUTOMATION:
            raise ValueError("automation_test_names only valid when editor_access=editor_automation")
        return v


class Dependency(StrictBase):
    task_id: _ID
    required_state: Literal["PASSED"] = "PASSED"


# ----------------------------- Top-level manifest -----------------------------

class TaskManifest(StrictBase):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    task_id: _ID
    title: str = Field(min_length=1, max_length=200)
    execution_class: ExecutionClass
    objective: str = Field(min_length=1, max_length=4000)
    constraints: Constraints = Field(default_factory=Constraints)
    acceptance_criteria: list[str] = Field(default_factory=list)
    repo: Repo
    paths: PathPolicy = Field(default_factory=PathPolicy)
    commands: CommandPolicy = Field(default_factory=CommandPolicy)
    model_profile: ModelProfile = Field(default_factory=ModelProfile)
    limits: Limits = Field(default_factory=Limits)
    context_budget: ContextBudget = Field(default_factory=ContextBudget)
    dependencies: list[Dependency] = Field(default_factory=list)
    unreal: Unreal = Field(default_factory=Unreal)

    @model_validator(mode="after")
    def _cross(self) -> "TaskManifest":
        # source_mutation / unreal_editor must be the ones allowed to declare writes
        cls = self.execution_class
        if cls == ExecutionClass.READ_ONLY and self.paths.write_paths:
            raise ValueError("read_only cannot declare write_paths")
        if cls == ExecutionClass.READ_ONLY and self.paths.create_paths:
            raise ValueError("read_only cannot declare create_paths")
        if cls == ExecutionClass.READ_ONLY and self.paths.protected_read_paths:
            # protected reads are technically allowed for read_only but
            # make the user opt in by listing them. Keep allowed.
            pass
        # unreal_editor tasks are blocked at execution time; allow manifest but
        # broker preflight will force BLOCKED.
        if cls == ExecutionClass.UNREAL_EDITOR and self.unreal.editor_access == EditorAccess.NONE:
            raise ValueError("unreal_editor execution_class requires unreal.editor_access != none")
        # editor_automation tasks are blocked at execution time. The manifest
        # itself is allowed so users can author them; runner forces BLOCKED.
        if self.unreal.editor_access == EditorAccess.EDITOR_AUTOMATION and cls != ExecutionClass.UNREAL_EDITOR:
            raise ValueError("editor_access=editor_automation requires execution_class=unreal_editor")
        # No wildcard writes (forbid * and ?)
        for lst in (self.paths.write_paths, self.paths.create_paths, self.paths.read_paths, self.paths.protected_read_paths):
            for p in lst:
                if any(ch in p for ch in ("*", "?", "[", "]")):
                    raise ValueError(f"wildcards not allowed in paths: {p!r}")
                if p.startswith("/") or ".." in p.split("/"):
                    raise ValueError(f"absolute or traversal path not allowed: {p!r}")
        return self


# ----------------------------- Patch payloads -----------------------------

class ReadExactArgs(StrictBase):
    tool: Literal["read_exact"] = "read_exact"
    path: str
    start_line: int | None = Field(default=None, ge=0)
    end_line: int | None = Field(default=None, ge=0)


class ReplaceExactArgs(StrictBase):
    tool: Literal["propose_patch"] = "propose_patch"
    op: Literal["replace_exact"] = "replace_exact"
    path: str
    expected_sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    old_text: str = Field(min_length=1, max_length=200_000)
    new_text: str = Field(min_length=0, max_length=200_000)
    expected_occurrences: int = Field(default=1, ge=1, le=4096)


class ReplaceFileArgs(StrictBase):
    tool: Literal["propose_patch"] = "propose_patch"
    op: Literal["replace_file"] = "replace_file"
    path: str
    expected_sha256: Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
    new_content: str = Field(min_length=0, max_length=1_000_000)


class CreateFileArgs(StrictBase):
    tool: Literal["propose_patch"] = "propose_patch"
    op: Literal["create_file"] = "create_file"
    path: str
    new_content: str = Field(min_length=0, max_length=1_000_000)


class ApplyValidatedPatchArgs(StrictBase):
    tool: Literal["apply_validated_patch"] = "apply_validated_patch"
    proposal_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]


class RunCommandArgs(StrictBase):
    tool: Literal["run_command_id"] = "run_command_id"
    command_id: _ID


class ReportResultArgs(StrictBase):
    tool: Literal["report_result"] = "report_result"
    disposition: Disposition
    summary: str = Field(default="", max_length=4000)
    evidence: list[str] = Field(default_factory=list)

    @field_validator("evidence", mode="before")
    @classmethod
    def _coerce_evidence(cls, v):
        # Models occasionally send a single string or None; coerce to list[str].
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]


class ToolCall(StrictBase):
    """One tool invocation. Strict union via discriminator."""
    call_id: str = Field(min_length=1, max_length=128)
    args: ReadExactArgs | ReplaceExactArgs | ReplaceFileArgs | CreateFileArgs | ApplyValidatedPatchArgs | RunCommandArgs | ReportResultArgs


__all__ = [
    "SCHEMA_VERSION",
    "ExecutionClass", "EditorAccess", "Disposition", "TaskStatus",
    "canonical_json", "sha256_bytes", "canonical_sha",
    "TaskManifest",
    "ToolCall",
    "ReadExactArgs", "ReplaceExactArgs", "ReplaceFileArgs", "CreateFileArgs",
    "ApplyValidatedPatchArgs", "RunCommandArgs", "ReportResultArgs",
]
