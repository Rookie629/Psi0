# Sonic Body29 Gripper Finetune Change Log

Date: 2026-05-01

## Goal

Add a GR00T N1.6 finetune entry point for Sonic-style full-body control while keeping the input/output surface simple:

- Use the 29 body DoF as state and action.
- Add left and right gripper-like hand channels.
- Ignore Sonic encoder inputs, decoder inputs, token state, and history-frame vectors for now.
- Do not use `base_height_command` or `navigate_command`.

This document records the experiment-level changes only. It does not describe a final Sonic deployment policy interface.

## Changed Files

### `src/gr00t/gr00t/configs/modality/g1_sonic_body29_vla.py`

Adds a modality config registered to `EmbodimentTag.NEW_EMBODIMENT`.

The config reads:

```text
${DATASET_PATH}/meta/modality.json
```

at import time, validates the required modality keys, and builds the GR00T modality mapping.

Current key order:

```text
left_leg
right_leg
waist
left_arm
left_hand
right_arm
right_hand
```

The gripper-like channels are derived from `meta/info.json`:

- `observation.state` has no separate `left_gripper` or `right_gripper` field.
- `action` has no separate `left_gripper` or `right_gripper` field.
- Both state and action store hand control as seven left-hand joints and seven right-hand joints.
- `observation.eef_state` and `action.eef` only contain wrist pose and quaternion fields, so they are not used as gripper state/action.

With the current local dataset, the resolved keys are:

```text
state_keys:
left_leg, right_leg, waist, left_arm, left_hand, right_arm, right_hand

action_keys:
left_leg, right_leg, waist, left_arm, left_hand, right_arm, right_hand
```

The resolved dimensions on `datasets/wbc_g1_pnp_no_ep0` are:

```text
state_dim = 43
action_dim = 43
```

This is 29 body DoF plus the two 7D hand groups from the current dataset. A strict 31D layout requires true 1D `left_gripper` and `right_gripper` fields in the dataset metadata and parquet rows, or a dataset conversion step that derives those fields from the 7D hand vectors.

### `baselines/gr00t-n1.6/presets/train/finetune_sonic_body29.yaml`

Adds a preset that points GR00T finetuning at the Sonic body29 gripper modality config.

Important fields:

```yaml
dataset:
  path: /home/yangke/KY/Psi0/datasets/wbc_g1_pnp_no_ep0
  embodiment_tag: NEW_EMBODIMENT
  modality_config_path: src/gr00t/gr00t/configs/modality/g1_sonic_body29_vla.py
training:
  output_dir: ./checkpoints/finetune_sonic_body29_gripper_gr00t
  max_steps: 10000
  save_steps: 1000
env:
  DATASET_PATH: /home/yangke/KY/Psi0/datasets/wbc_g1_pnp_no_ep0
  ACTION_HORIZON: "16"
```

The preset default is one GPU. Command-line overrides can set the GPU list and number of processes.

### `baselines/gr00t-n1.6/finetune_gr00t.py`

Updates launcher environment handling so `--dataset-path` also updates `DATASET_PATH`.

This matters because the modality config imports and validates `DATASET_PATH` before training starts. Without this synchronization, a command-line dataset override could train on one dataset path while validating modality metadata from a stale preset path.

## Run Command

For the user's target dataset:

```bash
src/gr00t/.venv/bin/python baselines/gr00t-n1.6/finetune_gr00t.py \
  --preset finetune_sonic_body29 \
  --cuda-visible-devices 0,1,2 \
  --num-gpus 3 \
  --dataset-path /root/ky/Psi0/WBC_VLA_data/G1_PnP
```

For the local test dataset, omit `--dataset-path`:

```bash
src/gr00t/.venv/bin/python baselines/gr00t-n1.6/finetune_gr00t.py \
  --preset finetune_sonic_body29 \
  --cuda-visible-devices 0,1,2 \
  --num-gpus 3
```

## Verified Checks

The following checks were run against the local dataset:

```bash
DATASET_PATH=/home/yangke/KY/Psi0/datasets/wbc_g1_pnp_no_ep0 \
PYTHONPATH=/home/yangke/KY/Psi0/src:/home/yangke/KY/Psi0/src/gr00t \
src/gr00t/.venv/bin/python -c "import src.gr00t.gr00t.configs.modality.g1_sonic_body29_vla as cfg; meta=cfg.MODALITY_META; keys=cfg.g1_sonic_body29_config['state'].modality_keys; akeys=cfg.g1_sonic_body29_config['action'].modality_keys; print('state_keys', keys); print('action_keys', akeys); print('state_dim', sum(meta['state'][k]['end']-meta['state'][k]['start'] for k in keys)); print('action_dim', sum(meta['action'][k]['end']-meta['action'][k]['start'] for k in akeys))"
```

Expected local output:

```text
state_keys ['left_leg', 'right_leg', 'waist', 'left_arm', 'left_hand', 'right_arm', 'right_hand']
action_keys ['left_leg', 'right_leg', 'waist', 'left_arm', 'left_hand', 'right_arm', 'right_hand']
state_dim 43
action_dim 43
```

The three-GPU dry run also succeeds:

```bash
src/gr00t/.venv/bin/python baselines/gr00t-n1.6/finetune_gr00t.py \
  --preset finetune_sonic_body29 \
  --cuda-visible-devices 0,1,2 \
  --num-gpus 3 \
  --dry-run
```

The generated command uses:

```text
--nproc_per_node=3
--modality-config-path src/gr00t/gr00t/configs/modality/g1_sonic_body29_vla.py
--output-dir ./checkpoints/finetune_sonic_body29_gripper_gr00t
```

## Known Limits

- This experiment does not consume `observation.encoder_input`, `observation.decoder_input`, or `observation.token_state`.
- This experiment does not reconstruct Sonic history-frame observations.
- Current local data uses 7D `left_hand/right_hand`, so it is not a strict 31D body-plus-gripper layout.
- If strict 1D grippers are required, add `left_gripper/right_gripper` to the dataset metadata and data columns, or define a conversion rule from the 7D hand vectors before training.
- The preset uses `global_batch_size: 8`. For exactly one sample per GPU on three GPUs, change it to `3`.
