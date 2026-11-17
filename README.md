# predict_solvent_content

Machine learning and analytical baselines for predicting macromolecular crystal solvent content (%) from X-ray diffraction normalized Patterson ($E^2$) maps.

## End-to-End Pipeline

1. **Generate Centered, Normalized Patterson Grids (`generate_maps.py`)**
   Computes normalised $E^2$ Patterson maps with `gemmi` from `.mtz`, `.cif`, or `.map(.gz)` files, applies a uniform resolution cutoff (`--d-min 3.5` Å), masks periodic & Bravais-centering origin spheres (`r < 4.0` Å), Z-score normalizes non-origin density, and interpolates onto a centered `128³` (`1.0` Å spacing) `float16` Cartesian grid (`4 MB` per map) alongside `map_metadata.tsv`:
   ```bash
   python generate_maps.py sf_files.txt --output-dir maps_128 --metadata-out map_metadata.tsv
   ```

2. **Cluster at <30% Sequence Identity & Run Baselines (`split_and_baselines.py`)**
   Groups homologous proteins (`<30%` sequence identity + crystal-form grouping), writes stratified `training_data.txt`, `validation_data.txt`, and `test_data.txt`, and benchmarks **Baseline A** (metadata-only ML + $d_{\min}$ confounder check) and **Baseline B** (analytical Matthews / Kantardjieff & Rupp probability):
   ```bash
   python split_and_baselines.py --results results.tsv --fasta curated_sequences.fasta --map-metadata map_metadata.tsv --map-dir maps_128
   ```

3. **Extract 3D Patterson + Pre-Phasing Diffraction Descriptors & Train Tree Ensembles (`generate_descriptors.py`, `train_descriptors.py`)**
   Computes 70 features, combining 55 3D Patterson descriptors (unit-cell/ASU spike-winsorized moments, spherical radial-shell profiles across 60 Å, intermolecular-to-intramolecular ratios, 3D EDT void/flat-channel radii, and 3D ridge skeleton branch density) with 15 standard pre-phasing diffraction/indexing features (native_d_min, d_max, Wilson B-factor, low-res Babinet intensity ratio, sorted cell lengths, asu_equiv_radius, space group), and benchmarks Random Forest, `HistGradientBoostingRegressor` (L2 & robust L1), and LightGBM (if installed):
   ```bash
   python generate_descriptors.py --input sf_files.txt --output additional_descriptors.txt --workers 8
   python train_descriptors.py training_data.txt validation_data.txt --descriptors additional_descriptors.txt --map-metadata map_metadata.tsv --test-file test_data.txt --output-model descriptor_model.joblib
   ```

4. **Train Multi-Modal 3D Deep Models, Export & Run Unified Inference (`train.py`, `export_to_onnx.py`, `predict_solvent_content.py`)**
   Trains `ResidualUNetSE3D` (`unet`) or `ViTVNet` (`vitvnet`) with 48-way 3D cube-symmetry augmentations, pre-phasing metadata FiLM fusion (`--map-metadata`), Weighted Huber loss with Inverse Label-Density Weighting, CUDA mixed precision (`torch.amp`), and `AdamW` + cosine warmup (supporting multi-GPU `torchrun` DDP):
   ```bash
   python train.py training_data.txt validation_data.txt unet --map-metadata map_metadata.tsv
   python train.py training_data.txt validation_data.txt vitvnet --map-metadata map_metadata.tsv
   python predict_solvent_content.py example.mtz models/best_unet.pt --gpu
   python predict_solvent_content.py example.map descriptor_model.joblib
   python export_to_onnx.py models/best_unet.pt unet --output unet_model.onnx
   ```

## Running Tests

```bash
python -m unittest discover -s tests -v
```
