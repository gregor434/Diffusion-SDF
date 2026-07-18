# Diffusion-SDF: Conditional Generative Modeling of Signed Distance Functions

[**Paper**](https://arxiv.org/abs/2211.13757) | [**Supplement**](https://light.princeton.edu/wp-content/uploads/2023/03/diffusionsdf_supp.pdf) | [**Project Page**](https://light.princeton.edu/publication/diffusion-sdf/) <br>

This repository contains the official implementation of <br> 
**[ICCV 2023] Diffusion-SDF: Conditional Generative Modeling of Signed Distance Functions** <br>
[Gene Chou](https://genechou.com), [Yuval Bahat](https://sites.google.com/view/yuval-bahat/home), [Felix Heide](https://www.cs.princeton.edu/~fheide/) <br>


If you find our code or paper useful, please consider citing
```bibtex
@inproceedings{chou2022diffusionsdf,
title={Diffusion-SDF: Conditional Generative Modeling of Signed Distance Functions},
author={Gene Chou and Yuval Bahat and Felix Heide},
journal={The IEEE International Conference on Computer Vision (ICCV)},
year={2023}
}
```


```cpp
root directory
  ├── config  
  │   └── // folders for checkpoints and training configs
  ├── data / datasets  
  │   └── // preprocessed SDF csv files, grid csv files, and split manifests (json)
  ├── models  
  │   ├── // models and lightning modules; main model is 'combined_model.py'
  │   └── archs
  │       └── // architectures such as PointNets, SDF MLPs, diffusion network..etc
  ├── dataloader  
  │   └── // dataloaders for different stages of training and generation
  ├── utils  
  │   └── // reconstruction and evaluation
  ├── metrics  
  │   └── // reconstruction and evaluation
  ├── diff_utils  
  │   └── // helper functions for diffusion
  ├── environment.yml  // package requirements
  ├── train.py  // script for training, specify the stage of training in the config files
  ├── test.py  // script for testing, specify the stage of testing in the config files
  └── tensorboard_logs  // created when running any training script
  
```

## Installation
We recommend creating an [anaconda](https://www.anaconda.com/) environment using our provided `environment.yml`:

```
conda env create -f environment.yml
conda activate diffusionsdf
```

## Dataset
For training, we preprocess all meshes and store query coordinates and signed distance values in csv files. Each csv file corresponds to one object, and each line represents a coordinate followed by its signed distance value. See `data/acronym` for examples. Modify the dataloader according to your file format. <br>

When sampling query points, make sure to also **sample uniformly within the 3D grid space** (i.e. from (-1,-1,-1) to (1,1,1)) rather than only sampling near the surface to avoid artifacts. For each training batch, we take 70% of query points sampled near the object surface and 30% sampled uniformly in the grid. `grid_source` in our dataloader and config file refers to the latter. <br>

The loaders expect split files with the following JSON shape:

```json
{
  "abo": {
    "ABO": [
      "3dmodel_id_1",
      "3dmodel_id_2"
    ]
  }
}
```

The preprocessed directory layout used by the ABO scripts is:

```text
datasets/
  abo/
    ABO/
      <3dmodel_id>/
        sdf_data.csv
  grid_data/
    abo/
      ABO/
        <3dmodel_id>/
          grid_gt.csv
  splits/
    abo_all_all.json
    abo_all_train.json
    abo_all_val.json
    abo_<product_type>_all.json
    abo_<product_type>_train.json
    abo_<product_type>_val.json
    abo_metadata.json
```

`scripts/prepare_abo_dataset.py` writes balanced aggregate splits in `datasets/splits/abo_all_{train,val}.json` by downsampling every `product_type_key` to the smallest category before splitting. It also writes per-category manifests under `datasets/splits/abo_<product_type>_{train,val}.json`. By default, `--train-ratio 0.8` produces a simple `80/20` `train/val` split, and the output is deterministic for a fixed `--seed`.

The preprocessing defaults match the paper: meshes are centered and uniformly scaled so the diagonal of the tight bounding box has length 1; 235,000 surface points are stored with zero SDF; two isotropic Gaussian query sets with standard deviations 0.005 and 0.0005 are generated from those points and evaluated against the mesh; and a regular 128 x 128 x 128 grid spanning `[-1, 1]^3` is stored. Existing CSV files are overwritten unless `--skip-existing` is supplied. To regenerate only the objects in an existing split, pass its path with `--only-models-in`.

For example, this replaces the derived data for the ABO chair split used by `config/abo/stage1_sdf/specs.json` while retaining all existing split definitions:

```bash
python scripts/prepare_abo_dataset.py --only-models-in datasets/splits/abo_CHAIR_all.json
```

ABO meshes are often not watertight. For categories such as chairs, the default non-watertight fallback signs SDF samples with the closest triangle normal, which can create incorrect negative regions in the uniform grid. To generate SDF labels from a watertight proxy while still sampling the training point cloud from the original cleaned mesh, build [ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus) outside this repository. `environment.yml` includes the generic build tools (`git`, `cmake`, `make`, and a Linux C++ compiler), but the ManifoldPlus checkout and build output should stay untracked because the build is machine-specific:

```bash
git clone --recursive https://github.com/hjwdzh/ManifoldPlus /path/to/ManifoldPlus
cd /path/to/ManifoldPlus
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

cd /path/to/Diffusion-SDF
export MANIFOLDPLUS_BIN=/path/to/ManifoldPlus/build/ManifoldPlus
python scripts/prepare_abo_dataset.py \
  --only-models-in datasets/splits/abo_CHAIR_all.json \
  --repair-method manifoldplus \
  --manifoldplus-depth 8
```

The script resolves the executable from `--manifoldplus-bin`, then `MANIFOLDPLUS_BIN`, then `PATH`. The repaired OBJ proxies are cached under `datasets/repaired_meshes` by default. Use `--force-repair` to regenerate them, `--repaired-mesh-dir` to choose a different cache location, and increase `--manifoldplus-depth` only after visual inspection if thin chair parts are over-smoothed. ManifoldPlus is external C++/CMake software with its own license terms, including non-commercial-use language in its README.

ABO training configs can point `TrainSplit` and `TestSplit` directly at these manifest files, for example `datasets/splits/abo_all_train.json` and `datasets/splits/abo_all_val.json`.

Image-conditioned diffusion training prepares cached CLIP features in `train.py` before DataLoader workers start, then reads those cached CPU tensors from the dataloader so workers do not initialize CUDA.

## Training
As described in our [paper](https://arxiv.org/abs/2211.13757), there are three stages of training. All corresponding config files can be found in the `config` folders. Logs are created in a `tensorboard_logs` folder in the root directory. We recommend tuning the `"kld_weight"` when training the joint SDF-VAE model as it enforces the continuity of the latent space. A higher value (e.g. 0.1) will result in better interpolation and generalization but sometimes more artifacts. A lower value (e.g. 0.00001) will result in worse interpolation but higher quality of generations. <br>

1. Training SDF modulations

```
python train.py -e config/stage1_sdf/ -b 32 -w 8    # -b for batch size, -w for workers, -r to resume training
```
Training notes: For Acronym / ShapeNet datasets, the loss should go down to $6 \sim 8 \times 10^{-4}$. Run testing to visualize whether the quality of reconstructed shapes is sufficient. The quality of reconstructions will carry over to the quality of generations. Note that the dimension of the VAE latent vectors will be 3 times `"latent_dim"` in `"SdfModelSpecs"` listed in the config file.

2. Training the diffusion model using the modulations extracted from the first stage 

```
# extract the modulations / latent vectors, which will be saved in a "modulations" folder in the config directory
# the folder needs to correspond to "data_path" in the diffusion config files

python test.py -e config/stage1_sdf/ -r last

# unconditional
python train.py -e config/stage2_diff_uncond/ -b 32 -w 8 

# conditional
python train.py -e config/stage2_diff_cond/ -b 32 -w 8 
```
Training notes: When extracting modulations, we recommend filtering based on the chamfer distance. See `test_modulations()` in `test.py` for details. Some notes on the conditional config file:  `"perturb_pc":"partial"`, `"crop_percent":0.5`, and `"sample_pc_size":128` refers to cropping 50% of a point cloud with 128 points to use as condition. `dim` in `diffusion_model_specs` needs to be the dimension of the latent vector, which is 3 times `"latent_dim"` in `"SdfModelSpecs"`. <br>


3. End-to-end training using the saved models from above 

```
# unconditional
python train.py -e config/stage3_uncond/ -b 32 -w 8 -r finetune     # training from the saved models of first two stages
python train.py -e config/stage3_uncond/ -b 32 -w 8 -r last     # resuming training if third stage has been trained 

# conditional
python train.py -e config/stage3_cond/ -b 32 -w 8 -r finetune    # training from the saved models of first two stages
python train.py -e config/stage3_cond/ -b 32 -w 8 -r last     # resuming training if third stage has been trained 
```
Training notes: The config file needs to contain the saved checkpoints for the previous two stages of training. The sdf loss (not generated sdf loss) should approach $6 \sim 8 \times 10^{-4}$.

## Testing
1. Testing SDF reconstructions and saving modulations

After the first stage of training, visualize / test reconstructions and save modulations:
```
# extract the modulations / latent vectors, which will be saved in a "modulations" folder in the config directory
# the folder needs to correspond to "data_path" in the diffusion config files
python test.py -e config/stage1_sdf/ -r last
```
A `recon` folder in the config directory will contain the `.ply` reconstructions and a `cd.csv` file that logs Chamfer Distance (CD). A `modulation` folder will contain `latent.txt` files for each SDF. The `modulation` folder will be the data path to the second stage of training.

2. Generations 

Meshes can be generated after the second or third stage of training.
```
python test.py -e config/stage3_uncond/ -r finetune  # generation after second stage 
python test.py -e config/stage3_uncond/ -r last      # after third stage 
```
A `recon` folder in the config directory will contain the `.ply` reconstructions. `max_batch` arguments in `test.py` are used for running marching cubes; change it to the max value your GPU memory can hold.


## References
We adapt code from <br>
GenSDF https://github.com/princeton-computational-imaging/gensdf <br>
DALLE2-pytorch https://github.com/lucidrains/DALLE2-pytorch <br>
Convolutional Occupancy Networks https://github.com/autonomousvision/convolutional_occupancy_networks (for PointNet encoder) <br>
Multimodal Shape Completion via cGANs https://github.com/ChrisWu1997/Multimodal-Shape-Completion (for conditional metrics) <br>
PointFlow https://github.com/stevenygd/PointFlow (for unconditional metrics)
