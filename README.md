# Diffusion-SDF with COD-VAE

This branch replaces Diffusion-SDF's PointNet/global-VAE representation with
[COD-VAE](https://github.com/join16/COD-VAE), while retaining the original
three-stage training workflow, Lightning checkpoints, JSON specification files,
marching-cubes reconstruction, conditioning, and shape metrics.

The representation path is:

    surface points [B,N,3]
      -> COD point/patch encoder
      -> diagonal posterior and COD tokens [B,M,D]
      -> COD latent decoder
      -> COD tri-planes [B,3,C,R,R]
      -> original Diffusion-SDF neural SDF head
      -> signed distances [B,Q]

The official COD plane order, axis projection, bilinear interpolation, and sum
fusion are unchanged. COD's occupancy and uncertainty heads remain instantiated
so official weights load strictly, but the recommended SDF profile does not use
them. It refines every spatial token and adds the residual planes directly to
the decoder's initial planes.

## Installation

    conda env create -f environment.yml
    conda activate diffusionsdf

COD-VAE's CUDA pointops extension is optional. When unavailable, the runtime
uses a checkpoint-neutral PyTorch farthest-point-sampling fallback.

Store downloaded reference repositories and checkpoints only in this
workspace's ignored `tmp/` directory. The included m32 profiles expect
`tmp/cod_vae_m32_weights.pt`. Download either official vae_m32 or vae_m64
weights from the
[COD-VAE weight folder](https://drive.google.com/drive/folders/1aJE_LbnyV8lBqjRc7tcXjui52N1Kvvjm)
and set CODVaeSpecs.checkpoint_path. The matching latent_tokens value must be
32 or 64. The vendored runtime and provenance notes are in models/cod_vae/.

## Data preprocessing

Download the complete filtered ABO chair set and its main catalog images:

    python scripts/download_abo_chairs.py \
      --target-root datasets/ABO/models_chair_full \
      --metadata-out datasets/ABO/abo_chairs_full.json \
      --skip-existing \
      --max-workers 16

    python datasets/ABO/scripts/download_abo_images.py \
      --subset-json datasets/ABO/abo_chairs_full.json \
      --models-dir datasets/ABO/models_chair_full \
      --target-root datasets/ABO/images_filtered \
      --main-image-only \
      --skip-existing \
      --max-workers 16

Preprocess the full chair set under a separate split prefix. This writes
`abo_fullchairs_CHAIR_{all,train,val}.json` and leaves the original
`abo_CHAIR_{all,train,val}.json` manifests unchanged:

    MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 python scripts/prepare_abo_dataset.py \
      --source-dir datasets/ABO/models_chair_full \
      --metadata-in datasets/ABO/abo_chairs_full.json \
      --metadata-out datasets/abo/fullchairs_preprocessing_metadata.json \
      --split-prefix abo_fullchairs \
      --per-type-splits-only \
      --skip-existing \
      --batch-size 50000 \
      --continue-on-error \
      --repair-method manifoldplus \
      --manifoldplus-bin tmp/ManifoldPlus/build/manifold \
      --repaired-mesh-dir datasets/repaired_meshes_cod_0999 \
      --manifoldplus-depth 8 \
      --repair-fidelity-samples 20000 \
      --repair-fidelity-max-p95 0.02 \
      --repair-fidelity-distance-threshold 0.02 \
      --repair-fidelity-max-outlier-fraction 0.05

ABO preprocessing writes one record per object:

    datasets/<dataset>/<class>/<object>/cod_sdf.npz

Each accepted record contains surface_points, surface_normals,
near_surface_query_points, near_surface_sdf, uniform_query_points, uniform_sdf,
normalization_center, normalization_scale, repaired-surface provenance, and
repair-fidelity measurements. With ManifoldPlus enabled, all surface samples,
normals, query points, and SDF targets come from the repaired watertight mesh.

All coordinates use one isotropic transform:

    x' = normalization_scale * (x - normalization_center)
    d' = normalization_scale * d

The tight bounding box is centered and scaled to a maximum absolute coordinate
of 0.999, matching COD's [-1,1] query domain. SDF values are evaluated after
this transform, so they already have the correctly scaled units.

    python scripts/prepare_abo_dataset.py \
      --only-models-in datasets/splits/abo_CHAIR_all.json \
      --surface-point-count 235000 \
      --uniform-point-count 262144

For non-watertight ABO meshes, the existing ManifoldPlus path remains:

    python scripts/prepare_abo_dataset.py \
      --only-models-in datasets/splits/abo_CHAIR_all.json \
      --repair-method manifoldplus \
      --manifoldplus-bin tmp/ManifoldPlus/build/manifold \
      --repaired-mesh-dir datasets/repaired_meshes_cod_0999 \
      --manifoldplus-depth 8

COD-normalized repaired proxies are cached separately under
datasets/repaired_meshes_cod_0999. Do not reuse datasets/repaired_meshes: that
legacy cache was generated after diagonal normalization and is in a different
coordinate system. A bidirectional sampled-surface comparison rejects repaired
meshes that differ too much from the normalized source mesh. Rejected objects
are recorded in preprocessing metadata and omitted before train/validation
splits are created; rejection is not treated as a processing crash. Per-proxy
fidelity sidecars make interrupted runs resumable.

`--skip-existing` only reuses records whose `surface_source` is
`repaired_mesh` and whose stored fidelity passes the current thresholds. Legacy
records whose COD points came from the original mesh are regenerated. The
dataloader contains no repair policy: it reads only IDs admitted to the emitted
manifests. SurfacePointCount, SampPerMesh, and NearSurfaceRatio remain JSON spec
fields. Preprocessing metadata is written to
datasets/abo/preprocessing_metadata.json; datasets/splits remains reserved for
split manifests.

To regenerate `cod_sdf.npz` records while reusing both the cached repaired
meshes and their compatible fidelity validation sidecars, omit
`--skip-existing` and pass `--reuse-repair-fidelity`. A missing sidecar or one
created with a different fidelity sample count or distance threshold is
validated again automatically. `--force-repair` still regenerates the proxy and
therefore always performs a fresh fidelity validation.

## Stage one: COD-VAE SDF reconstruction

    python train.py -e config/cod/stage1_frozen_pretrained -b 8 -w 8

Adapt the COD latent and tri-plane decoders from the further-trained
head-refinement checkpoint while keeping the point encoder and posterior fixed:

    python train.py \
      -e config/cod/stage1_decoder_finetune \
      --init_from config/cod/stage1_head_refine/last.ckpt \
      -b 8 -w 8

For the single-object overfit profile, a virtual training size keeps batches
full while each repeated access independently resamples surface and SDF query
points:

    python train.py -e config/cod/stage1_overfit_one -b 10 -w 8 --virtual_train_size 100

Available stage1_mode values are sdf_head_only, cod_decoder_finetune,
full_cod_finetune, and train_from_scratch.

learning_rates accepts independent values for point_encoder, variational_block,
latent_decoder, triplane_decoder, and sdf_network, and rejects entries for
components frozen by the selected mode. sdf_loss.type supports l1, huber, and
truncated_sdf. The recommended decoder adaptation uses truncated SDF and
decoded-token reconstruction losses; its initial-SDF, uncertainty, geometry,
normal, and KL losses are disabled.

Extract native, unflattened modulations with the existing command:

    python test.py -e config/cod/stage1_decoder_finetune -r last

Each modulation.npz stores object_id, posterior_mean [M,D], and
posterior_logvar [M,D]. Extraction uses the posterior mean and writes
per-channel training statistics to modulations/latent_stats.npz.

## Stage two: COD token diffusion

    python train.py -e config/cod/stage2_transformer_diffusion -b 64 -w 8

At startup, stage two caches modulations for the union of every configured
TrainSplit, ValSplit, TestSplit, and ModulationSplit under the stage-two
experiment directory at `modulations/`. Existing files are reused, and the
stage-one encoder is loaded only when a modulation is missing. Latent statistics
are computed from TrainSplit only. `modulation_batch_size` and
`modulation_workers` optionally control this one-time extraction (defaults: up
to 8 objects per batch and the training worker count). Stage two then reads the
cached modulation files and optional cached conditions.
The denoiser is a non-causal token transformer over [B,M,D]. It uses the EDM
log-normal noise distribution, EDM preconditioning, weighted denoising loss,
and second-order Heun sampling referenced by COD-VAE through VecSet.

    python test.py -e config/cod/stage2_transformer_diffusion -r last -n 5

Sampling denormalizes tokens, loads the stage-one SDF/COD checkpoint, decodes
tri-planes, queries the SDF head, and runs marching cubes.

## Stage three: joint fine-tuning

    python train.py -e config/cod/stage3_full_joint -b 8 -w 8 -r finetune

Available stage3_mode values are diffusion_only, diffusion_and_sdf,
diffusion_and_cod_decoder, and full_joint_finetune.

The generated-SDF branch decodes the EDM model's estimated clean latent for the
sampled noise level. It does not run a reverse trajectory inside a training
batch. Coefficients are loss_weights.direct, loss_weights.diffusion,
loss_weights.generated, and loss_weights.kl.

## Configuration profiles

The primary and ablation profiles live under config/cod/:

    stage1_frozen_pretrained
    stage1_decoder_finetune
    stage1_full_finetune
    stage1_from_scratch
    stage2_transformer_diffusion
    stage3_diffusion_only
    stage3_full_joint

Existing config/example and config/abo specs use the same COD fields. No second
configuration system or representation backend selector is introduced.

## Evaluation

The existing marching-cubes and distribution metric implementations remain.
Stage-one reports include SDF reconstruction error, near/uniform error, sign
accuracy, Chamfer distance, F-score, normal consistency, mesh validity,
encoding/decoding time, and peak CUDA memory. Diffusion training logs EDM loss,
and conditional generation uses the same per-shape mesh metrics.

## References

- Gene Chou, Yuval Bahat, and Felix Heide, “Diffusion-SDF,” ICCV 2023.
- In Cho, Youngbeom Yoo, Subin Jeon, and Seon Joo Kim, “Representing 3D Shapes
  with 64 Latent Vectors for 3D Diffusion Models,” ICCV 2025.
- Biao Zhang et al., “3DShape2VecSet,” SIGGRAPH 2023.

Please follow the upstream repositories' citation and licensing requirements.
