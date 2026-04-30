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
    raise RuntimeError("DATASET_PATH must be set to load WBC VLA modality.json")
META_PATH = Path(DATASET_PATH) / "meta" / "modality.json"
if not META_PATH.exists():
    raise RuntimeError(f"Missing modality.json at {META_PATH}")

with META_PATH.open("r") as f:
    MODALITY_META = json.load(f)

STATE_KEYS = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
]
ACTION_KEYS = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
    "base_height_command",
    "navigate_command",
]
VIDEO_KEYS = ["ego_view"]
LANGUAGE_KEYS = ["annotation.human.action.task_description"]

missing_state = sorted(set(STATE_KEYS) - set(MODALITY_META.get("state", {})))
missing_action = sorted(set(ACTION_KEYS) - set(MODALITY_META.get("action", {})))
missing_video = sorted(set(VIDEO_KEYS) - set(MODALITY_META.get("video", {})))
missing_annotation = sorted(
    {key.removeprefix("annotation.") for key in LANGUAGE_KEYS}
    - set(MODALITY_META.get("annotation", {}))
)
if missing_state or missing_action or missing_video or missing_annotation:
    raise RuntimeError(
        "WBC VLA modality.json missing required keys: "
        f"state={missing_state}, action={missing_action}, "
        f"video={missing_video}, annotation={missing_annotation}"
    )

ACTION_HORIZON = int(os.environ.get("ACTION_HORIZON", "16"))
if ACTION_HORIZON <= 0:
    raise RuntimeError(f"ACTION_HORIZON must be > 0, got {ACTION_HORIZON}")

g1_wbc_vla_config = {
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

register_modality_config(g1_wbc_vla_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
