import json
import os
from pathlib import Path

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


DATASET_PATH = os.environ.get("DATASET_PATH")
if not DATASET_PATH:
    raise RuntimeError("DATASET_PATH must be set to load Sonic body29 modality.json")
META_PATH = Path(DATASET_PATH) / "meta" / "modality.json"
if not META_PATH.exists():
    raise RuntimeError(f"Missing modality.json at {META_PATH}")
INFO_PATH = Path(DATASET_PATH) / "meta" / "info.json"
if not INFO_PATH.exists():
    raise RuntimeError(f"Missing info.json at {INFO_PATH}")

with META_PATH.open("r") as f:
    MODALITY_META = json.load(f)
with INFO_PATH.open("r") as f:
    INFO_META = json.load(f)

BODY29_PREFIX_KEYS = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
]
BODY29_SUFFIX_KEYS = [
    "right_arm",
]
LEFT_GRIPPER_KEY = "left_hand"
RIGHT_GRIPPER_KEY = "right_hand"
VIDEO_KEYS = ["ego_view"]
LANGUAGE_KEYS = ["annotation.human.action.task_description"]


def _validate_hand_names(feature_key: str, hand_key: str, expected_prefix: str) -> None:
    feature = INFO_META.get("features", {}).get(feature_key, {})
    names = feature.get("names", [])
    hand_meta = MODALITY_META.get(feature_key.split(".")[-1], {}).get(hand_key)
    if hand_meta is None:
        raise RuntimeError(f"Missing {hand_key} in modality metadata for {feature_key}")
    hand_names = names[hand_meta["start"] : hand_meta["end"]]
    if len(hand_names) != 7 or any(not name.startswith(expected_prefix) for name in hand_names):
        raise RuntimeError(
            f"{feature_key} {hand_key} must map to seven {expected_prefix} joints, got {hand_names}"
        )


_validate_hand_names("observation.state", LEFT_GRIPPER_KEY, "left_hand_")
_validate_hand_names("observation.state", RIGHT_GRIPPER_KEY, "right_hand_")
_validate_hand_names("action", LEFT_GRIPPER_KEY, "left_hand_")
_validate_hand_names("action", RIGHT_GRIPPER_KEY, "right_hand_")


STATE_KEYS = [
    *BODY29_PREFIX_KEYS,
    LEFT_GRIPPER_KEY,
    *BODY29_SUFFIX_KEYS,
    RIGHT_GRIPPER_KEY,
]
ACTION_KEYS = [
    *BODY29_PREFIX_KEYS,
    LEFT_GRIPPER_KEY,
    *BODY29_SUFFIX_KEYS,
    RIGHT_GRIPPER_KEY,
]

# 29 body DoF plus the gripper-like hand controls defined in info.json. The current
# dataset stores these controls as seven left_hand joints and seven right_hand joints.
missing_state = sorted(set(STATE_KEYS) - set(MODALITY_META.get("state", {})))
missing_action = sorted(set(ACTION_KEYS) - set(MODALITY_META.get("action", {})))
missing_video = sorted(set(VIDEO_KEYS) - set(MODALITY_META.get("video", {})))
missing_annotation = sorted(
    {key.removeprefix("annotation.") for key in LANGUAGE_KEYS}
    - set(MODALITY_META.get("annotation", {}))
)
if missing_state or missing_action or missing_video or missing_annotation:
    raise RuntimeError(
        "Sonic body29 modality.json missing required keys: "
        f"state={missing_state}, action={missing_action}, "
        f"video={missing_video}, annotation={missing_annotation}"
    )

ACTION_HORIZON = int(os.environ.get("ACTION_HORIZON", "16"))
if ACTION_HORIZON <= 0:
    raise RuntimeError(f"ACTION_HORIZON must be > 0, got {ACTION_HORIZON}")

g1_sonic_body29_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=VIDEO_KEYS,
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=STATE_KEYS,
    ),
    "action": ModalityConfig(
        delta_indices=list(range(ACTION_HORIZON)),
        modality_keys=ACTION_KEYS,
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
            for _ in ACTION_KEYS
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=LANGUAGE_KEYS,
    ),
}

register_modality_config(g1_sonic_body29_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
