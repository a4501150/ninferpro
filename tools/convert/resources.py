"""Select final resource bytes and derive the tokenizer's public token domain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping

TEXT_RESOURCES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)
VISION_RESOURCES = ("preprocessor_config.json", "video_preprocessor_config.json")


def _check_qwen_vision_resource(
    value: dict, vision_config: Mapping, path: Path
) -> None:
    """A Qwen-family vision resource must agree with the vision config record."""
    for field, config_field in (
        ("patch_size", "patch_size"),
        ("temporal_patch_size", "temporal_patch_size"),
        ("merge_size", "spatial_merge_size"),
    ):
        if (
            type(value.get(field)) is not int
            or value[field] != vision_config[config_field]
        ):
            raise ValueError(
                f"{path}: {field} differs from Vision config"
            )


def token_domain(
    tokenizer: dict, config: dict, vocab_size: int
) -> tuple[int, tuple[int, ...]]:
    ids: dict[int, str] = {}
    special: dict[int, bool] = {}

    def add(index: object, content: object) -> None:
        if (
            type(index) is not int
            or not 0 <= index < vocab_size
            or not isinstance(content, str)
        ):
            raise ValueError(
                f"invalid tokenizer entry id={index!r}, content={content!r}"
            )
        if index in ids and ids[index] != content:
            raise ValueError(f"tokenizer resources disagree about token {index}")
        ids[index] = content

    def add_special(index: int, token: dict) -> None:
        flag = token.get("special", False)
        if type(flag) is not bool:
            raise ValueError(f"token {index}: special must be boolean")
        if index in special and special[index] != flag:
            raise ValueError(
                f"tokenizer resources disagree about token {index} special flag"
            )
        special[index] = flag

    model = tokenizer.get("model", {})
    vocabulary = model.get("vocab")
    if not isinstance(vocabulary, dict):
        raise ValueError("tokenizer model must provide a vocabulary mapping")
    for token, index in vocabulary.items():
        add(index, token)
    for token in tokenizer.get("added_tokens", []):
        add(token["id"], token["content"])
        add_special(token["id"], token)
    for raw_id, token in config.get("added_tokens_decoder", {}).items():
        index = int(raw_id)
        add(index, token["content"])
        add_special(index, token)
    if not ids or set(ids) != set(range(max(ids) + 1)):
        raise ValueError("this tokenizer requires a contiguous public token ID domain")
    return len(ids), tuple(sorted(index for index, flag in special.items() if flag))


def load_resources(
    model_dir: Path,
    *,
    vocab_size: int,
    vision_config: Mapping[str, int] | None = None,
    vision_resources: tuple[str, ...] = VISION_RESOURCES,
    vision_check: Callable[[dict, Mapping, Path], None] | None = None,
    overrides: Mapping[str, str | Path] | None = None,
    normalize: Callable[[dict[str, dict], dict[str, bytes]], None] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, bytes], int, tuple[int, ...]]:
    """Read the selected components' resource files and derive the token domain.

    ``normalize`` is a family's tokenizer invariants (e.g.
    qwen_common.normalize_tokenizer): it may rewrite the parsed JSON resources and
    replace their stored payloads before the token domain is derived; this loader
    itself knows no family. ``vision_resources`` names the resource files of the
    vision component and ``vision_check`` validates each parsed one against the
    vision config record (default: the Qwen preprocessor check, applied to HF's
    separate preprocessor/video preprocessor files; a family shipping a single
    processor file supplies its own names and check).
    """
    overrides = {} if overrides is None else dict(overrides)
    roles = {"text": TEXT_RESOURCES}
    if vision_config is not None:
        roles["vision"] = vision_resources
    allowed = {role for names in roles.values() for role in names}
    if overrides.keys() - allowed:
        raise ValueError(
            f"resource overrides have no selected consumer: {sorted(overrides.keys()-allowed)}"
        )
    references: dict[str, dict[str, str]] = {}
    payloads: dict[str, bytes] = {}
    parsed: dict[str, dict] = {}
    check = vision_check if vision_check is not None else _check_qwen_vision_resource
    for component, names in roles.items():
        references[component] = {}
        for role in names:
            path = Path(overrides[role]) if role in overrides else model_dir / role
            data = path.read_bytes()
            if not data:
                raise ValueError(f"{path}: resource is empty")
            text = data.decode("utf-8")
            if role.endswith(".json"):
                value = json.loads(text)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}: resource must contain a JSON object")
                parsed[role] = value
                if component == "vision":
                    check(value, vision_config, path)
            object_id = f"resource/{component}/{role}"
            references[component][role] = object_id
            payloads[object_id] = data
    if normalize is not None:
        normalize(parsed, payloads)
    count, special = token_domain(
        parsed["tokenizer.json"], parsed["tokenizer_config.json"], vocab_size
    )
    return references, payloads, count, special
