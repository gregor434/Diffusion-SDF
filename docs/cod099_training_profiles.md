# COD 0.99 data and training profiles

The `cod099` datasets are isolated from the legacy `0.999` records. The
commands below deliberately select a new dataset key, split prefix, and
normalization extent. Split manifests are outputs of these commands; they
should not be created separately.

## Data preparation

Regenerate ABO records only from the accepted, deduplicated repaired proxies:

```bash
python scripts/prepare_abo_dataset.py \
  --accepted-proxy-mode \
  --accepted-proxy-dir datasets/repaired_meshes_cod_0999/abo/ABO \
  --source-all-manifest datasets/splits/abo_fullchairs_multiray21_CHAIR_all.json \
  --source-train-manifest datasets/splits/abo_fullchairs_multiray21_CHAIR_train.json \
  --source-val-manifest datasets/splits/abo_fullchairs_multiray21_CHAIR_val.json \
  --source-preprocessing-metadata datasets/abo_multiray21/fullchairs_preprocessing_metadata.json \
  --dataset-key abo_cod099 \
  --class-name ABO \
  --split-prefix abo_fullchairs_multiray21_cod099_CHAIR \
  --normalization-extent 0.99
```

The accepted-proxy mode does not read raw GLBs, invoke ManifoldPlus, evaluate
fidelity, deduplicate, or repartition. It writes
`datasets/abo_cod099/preprocessing_metadata.json` and the `all`, `train`, and
`val` manifests after the records have been generated.

Prepare ShapeNetCore and ShapeNetPart with explicit versioned outputs:

```bash
python scripts/prepare_shapenetcore_chairs.py \
  --dataset-key shapenetcore_cod099 \
  --split-prefix shapenetcore_cod099_CHAIR \
  --normalization-extent 0.99

python scripts/prepare_shapenetpart_chairs.py \
  --dataset-key shapenetpart_cod099 \
  --split-prefix shapenetpart_cod099_CHAIR \
  --normalization-extent 0.99
```

Omitting `--normalization-extent` retains the legacy `0.999` behavior.

## Training order

```bash
python scripts/train_cod099_stage1_pipeline.py --batch-size 16 --workers 8
python train.py -e config/cod/stage2_transformer_diffusion_shapenetcore_pretrain_fps_cod099
python train.py -e config/cod/stage2_fewshot_sdf_head_shapenetcore_fps_pretrained_finetune_cod099
python train.py -e config/cod/stage2_fewshot_sdf_head_shapenetcore_fps_pairwise_adaptation_cod099
```

The stage-one runner keeps the 0.99 checkpoint lineage isolated and executes
three experiments in order:

1. `stage1_sdf_head_multiray21_cod099` bootstraps only the SDF head at `2e-4`
   without eikonal or normal penalties.
2. `stage1_sdf_head_polish_multiray21_cod099` reloads the best bootstrap weights
   into a fresh `1e-4` optimizer for a short polishing pass.
3. `stage1_sdf_head_conv_refine_multiray21_cod099` enables the residual
   convolutional refiner and geometry regularization at `5e-5`/`2e-5`.

These settings follow the legacy 0.999 logs. Applying geometry penalties to a
random SDF head collapsed to a constant field with validation loss near `1.1`.
The geometry-free head reached `0.0283` near epoch 157, its lower-rate polish
reached `0.0262` near epoch 31, and convolutional refinement continued making
useful progress through epoch 767. The corresponding early-stopping patience
values are 30, 20, and 150 epochs. The runner also enforces validation-loss
quality gates of `0.05` after bootstrap and `0.04` after polishing.

Use the stage-specific `--skip-*` or `--resume-*` flags after an interruption.
Fresh stages refuse to start in directories containing checkpoints, and every
stage transition requires the preceding `best.ckpt`.

The ShapeNet pretraining profile owns its modulation cache. Both ABO
fine-tuning profiles share
`config/cod/stage2_fewshot_fps_cod099_shared/modulations`, including its
`latent_stats.npz`. Running either fine-tuning profile after stage one creates
that ABO cache if it is absent.
