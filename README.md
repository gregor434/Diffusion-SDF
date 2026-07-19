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
fusion are unchanged. COD's occupancy head remains instantiated so official
weights load strictly, but it is never used for SDF prediction. Uncertainty
pruning defaults to an effective keep ratio of 1 because its published head was
trained for occupancy.

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

ABO preprocessing writes one record per object:

    datasets/<dataset>/<class>/<object>/cod_sdf.npz

Each record contains surface_points, surface_normals,
near_surface_query_points, near_surface_sdf, uniform_query_points, uniform_sdf,
normalization_center, and normalization_scale.

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
coordinate system. The proxies are used for SDF signing; COD surface samples
still come from the normalized source mesh. SurfacePointCount, SampPerMesh, and
NearSurfaceRatio remain JSON spec fields. Preprocessing metadata is written to
datasets/abo/preprocessing_metadata.json; datasets/splits remains reserved for
split manifests.

## Stage one: COD-VAE SDF reconstruction

    python train.py -e config/cod/stage1_frozen_pretrained -b 8 -w 8

For the single-object overfit profile, a virtual training size keeps batches
full while each repeated access independently resamples surface and SDF query
points:

    python train.py -e config/cod/stage1_overfit_one -b 10 -w 8 --virtual_train_size 100

Available stage1_mode values are sdf_head_only, cod_decoder_finetune,
full_cod_finetune, and train_from_scratch.

learning_rates accepts independent values for point_encoder, variational_block,
latent_decoder, triplane_decoder, and sdf_network. sdf_loss.type supports l1,
huber, and truncated_sdf. Loss coefficients are loss_weights.sdf,
loss_weights.kl, and loss_weights.cod_aux.

Extract native, unflattened modulations with the existing command:

    python test.py -e config/cod/stage1_frozen_pretrained -r last

Each modulation.npz stores object_id, posterior_mean [M,D], and
posterior_logvar [M,D]. Extraction uses the posterior mean and writes
per-channel training statistics to modulations/latent_stats.npz.

## Stage two: COD token diffusion

    python train.py -e config/cod/stage2_transformer_diffusion -b 64 -w 8

Stage two reads only cached modulation files and optional cached conditions.
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

The seven requested profiles live under config/cod/:

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
