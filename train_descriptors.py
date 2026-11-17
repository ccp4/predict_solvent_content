#!/usr/bin/env python3
"""
Train, benchmark, inspect, and run tree-ensemble models (Random Forest,
HistGradientBoostingRegressor, and optional LightGBM) combining 3D Patterson
map descriptors with standard pre-phasing diffraction/indexing metadata
(d_min, unit cell, V_asu, space group, Wilson B, low-res intensity contrast).

Usage (Training):
    python train_descriptors.py training_data.txt validation_data.txt \
        --descriptors additional_descriptors.txt \
        --map-metadata map_metadata.tsv \
        --test-file test_data.txt \
        --output-model descriptor_model.joblib

Usage (Standalone Inference):
    python train_descriptors.py predict example.mtz --model descriptor_model.joblib
"""

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple
import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

from generate_descriptors import (
    DESCRIPTOR_COLUMNS,
    DIFFRACTION_METADATA_COLUMNS,
    PATTERSON_ONLY_COLUMNS,
    extract_descriptors_for_file,
)
from patterson import (
    DEFAULT_D_MIN,
    DEFAULT_GRID_SIZE,
    DEFAULT_GRID_SPACING,
    DEFAULT_ORIGIN_MASK_RADIUS,
)

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """Weighted blend of fitted regressors (e.g. GBDT + Random Forest)."""

    def __init__(
        self,
        estimators: List[BaseEstimator],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.estimators = estimators
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        self.fitted_estimators_ = []
        for est in self.estimators:
            cloned = clone(est)
            if sample_weight is not None:
                if hasattr(cloned, "steps"):
                    last_step = cloned.steps[-1][0]
                    cloned.fit(X, y, **{f"{last_step}__sample_weight": sample_weight})
                else:
                    cloned.fit(X, y, sample_weight=sample_weight)
            else:
                cloned.fit(X, y)
            self.fitted_estimators_.append(cloned)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.column_stack([est.predict(X) for est in self.fitted_estimators_])
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=np.float64)
            w = w / np.sum(w)
            return preds @ w
        return np.mean(preds, axis=1)


def load_descriptor_table(descriptor_path: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Load whitespace-delimited descriptor file written by `generate_descriptors.py`.
    Returns `(feature_names, {name_lower: feature_vector})`.
    """
    rows: Dict[str, np.ndarray] = {}
    feature_names: Optional[List[str]] = None

    with open(descriptor_path, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            parts = raw.strip().split()
            if not parts:
                continue
            if line_num == 1 and parts[0] == "name":
                feature_names = parts[1:]
                continue
            name = parts[0].lower()
            vals = np.array([float(x) for x in parts[1:]], dtype=np.float64)
            if feature_names is None:
                feature_names = list(DESCRIPTOR_COLUMNS[: len(vals)])
            if len(vals) == len(feature_names):
                rows[name] = vals

    if not feature_names:
        feature_names = list(DESCRIPTOR_COLUMNS)
    return feature_names, rows


def backfill_from_map_metadata(
    feature_names: List[str],
    desc_map: Dict[str, np.ndarray],
    map_metadata_path: Optional[Path],
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Optionally backfill missing (`NaN`) diffraction/indexing metadata columns from
    `map_metadata.tsv` (useful when descriptors were extracted from `.npy` maps).
    """
    if map_metadata_path is None or not map_metadata_path.is_file():
        return feature_names, desc_map

    # Ensure all standard diffraction metadata columns exist in feature_names
    added_cols = [c for c in DIFFRACTION_METADATA_COLUMNS if c not in feature_names]
    if added_cols:
        feature_names = list(feature_names) + added_cols
        for k, v in list(desc_map.items()):
            desc_map[k] = np.concatenate([v, np.full(len(added_cols), np.nan, dtype=np.float64)])

    col_idx = {c: i for i, c in enumerate(feature_names)}

    with open(map_metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            code = row["name"].strip().lower()
            vec = desc_map.get(code)
            if vec is None:
                continue

            def _get_f(key: str) -> float:
                val = row.get(key)
                if val is None or val == "":
                    return float("nan")
                try:
                    return float(val)
                except ValueError:
                    return float("nan")

            cell_vol = _get_f("cell_volume")
            asu_vol = _get_f("asu_volume")
            sg_num = _get_f("spacegroup_number")
            pg_ord = _get_f("point_group_order")
            candidates = {
                "origin_to_sigma": _get_f("origin_to_sigma"),
                "native_d_min": _get_f("native_d_min"),
                "d_max": _get_f("d_max"),
                "wilson_b_est": _get_f("wilson_b_est"),
                "low_res_intensity_ratio": _get_f("low_res_intensity_ratio"),
                "acentric_e2_minus_1": _get_f("acentric_e2_minus_1"),
                "mean_e4": _get_f("mean_e4"),
                "log_cell_volume": math.log(cell_vol) if (math.isfinite(cell_vol) and cell_vol > 1.0) else float("nan"),
                "log_asu_volume": math.log(asu_vol) if (math.isfinite(asu_vol) and asu_vol > 1.0) else float("nan"),
                "asu_equiv_radius": _get_f("asu_equiv_radius"),
                "cell_min_len": _get_f("cell_min_len"),
                "cell_mid_len": _get_f("cell_mid_len"),
                "cell_max_len": _get_f("cell_max_len"),
                "cell_anisotropy": _get_f("cell_anisotropy"),
                "spacegroup_number": sg_num if (math.isfinite(sg_num) and sg_num > 0) else float("nan"),
                "point_group_order": pg_ord if (math.isfinite(pg_ord) and pg_ord > 0) else float("nan"),
            }
            for cname, cval in candidates.items():
                idx = col_idx.get(cname)
                if idx is not None and not math.isfinite(vec[idx]) and math.isfinite(cval):
                    vec[idx] = cval

    return feature_names, desc_map


def select_feature_subset(
    feature_names: List[str],
    X_train: np.ndarray,
    feature_set: str,
) -> Tuple[List[str], List[int]]:
    if feature_set == "all":
        allowed = set(feature_names)
    elif feature_set == "patterson_only":
        allowed = set(PATTERSON_ONLY_COLUMNS)
    else:
        allowed = set(DIFFRACTION_METADATA_COLUMNS)

    # Retain columns in the requested subset that have at least 2 finite distinct values in X_train
    # (avoids scikit-learn HistGBR binning crash on 100%-NaN or constant columns)
    keep_idx: List[int] = []
    for i, col in enumerate(feature_names):
        if col not in allowed:
            continue
        finite_vals = X_train[np.isfinite(X_train[:, i]), i]
        if finite_vals.size >= 2 and float(np.ptp(finite_vals)) > 0.0:
            keep_idx.append(i)

    if not keep_idx:
        raise ValueError(f"No informative columns matched --feature-set={feature_set}")
    return [feature_names[i] for i in keep_idx], keep_idx


def load_split_targets(split_path: Path) -> List[Tuple[str, float]]:
    """Read `(name_lower, solvent_pct_in_0_100)` from a split file (`name solv path`)."""
    items: List[Tuple[str, float]] = []
    with open(split_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            name = parts[0].split(".")[0].lower()
            val = float(parts[1])
            solv_pct = val * 100.0 if val <= 1.0 else val
            items.append((name, solv_pct))
    return items


def align_split_features(
    split_items: List[Tuple[str, float]],
    desc_map: Dict[str, np.ndarray],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    names: List[str] = []
    X_rows: List[np.ndarray] = []
    y_vals: List[float] = []

    for name, solv_pct in split_items:
        vec = desc_map.get(name)
        if vec is None:
            continue
        clean_vec = np.where(np.isfinite(vec), vec, np.nan)
        names.append(name)
        X_rows.append(clean_vec)
        y_vals.append(solv_pct)

    if not X_rows:
        raise ValueError("No matching entries found between split file and descriptor table.")
    return names, np.vstack(X_rows), np.asarray(y_vals, dtype=np.float64)


def compute_sample_weights(
    y_pct: np.ndarray,
    mode: str = "inverse_density",
    tail_weight: float = 8.0,
    bandwidth_pct: float = 4.0,
) -> Optional[np.ndarray]:
    """
    Compute training sample weights to counter unimodal PDB solvent clustering (~47%):
      - 'inverse_density': Label Distribution Smoothing (1 / sqrt(KDE(y))), clipped to [0.5, 6.0].
      - 'linear_tail': `1 + tail_weight * |y/100 - 0.5|`, matching `WeightedMSE` in `train.py`.
      - 'none': uniform weights.
    """
    if mode == "none" or len(y_pct) < 4:
        return None
    if mode == "linear_tail":
        w = 1.0 + tail_weight * np.abs((y_pct / 100.0) - 0.5)
        return w / np.mean(w)

    diffs = (y_pct[:, None] - y_pct[None, :]) / max(0.5, bandwidth_pct)
    kde = np.mean(np.exp(-0.5 * diffs**2), axis=1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(kde, 1e-6))
    inv_sqrt /= np.mean(inv_sqrt)
    return np.clip(inv_sqrt, 0.5, 6.0)


def evaluate_predictions(
    y_true_pct: np.ndarray,
    y_pred_pct: np.ndarray,
    loss_weight: float = 8.0,
) -> Dict[str, float]:
    rmse = float(math.sqrt(mean_squared_error(y_true_pct, y_pred_pct)))
    mae = float(mean_absolute_error(y_true_pct, y_pred_pct))
    if len(y_true_pct) >= 2 and np.std(y_true_pct) > 1e-12 and np.std(y_pred_pct) > 1e-12:
        r = float(np.corrcoef(y_true_pct, y_pred_pct)[0, 1])
        r2 = float(r2_score(y_true_pct, y_pred_pct))
    else:
        r = float("nan")
        r2 = float("nan")

    yt01 = y_true_pct / 100.0
    yp01 = y_pred_pct / 100.0
    wmse = float(np.mean(((yp01 - yt01) ** 2) * (np.abs(yt01 - 0.5) * loss_weight + 1.0)))

    tail_mask = (y_true_pct < 35.0) | (y_true_pct > 65.0)
    tail_mae = float(mean_absolute_error(y_true_pct[tail_mask], y_pred_pct[tail_mask])) if np.any(tail_mask) else float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "tail_mae": tail_mae,
        "wmse": wmse,
        "r": r,
        "r2": r2,
    }


def build_candidate_models(n_train: int, seed: int = 42) -> Dict[str, BaseEstimator]:
    min_leaf_gbdt = max(2, min(20, n_train // 10))
    rf = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestRegressor(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=max(1, min(2, n_train // 10)),
            random_state=seed,
            n_jobs=-1,
        ),
    )
    hgb_l2 = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=min(31, max(3, n_train // 2)),
        min_samples_leaf=min_leaf_gbdt,
        l2_regularization=1.0,
        early_stopping=n_train >= 60,
        random_state=seed,
    )
    hgb_l1 = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=min(31, max(3, n_train // 2)),
        min_samples_leaf=min_leaf_gbdt,
        l2_regularization=1.0,
        early_stopping=n_train >= 60,
        random_state=seed,
    )

    models: Dict[str, BaseEstimator] = {
        "RandomForest": rf,
        "HistGBR (SquaredError)": hgb_l2,
        "HistGBR (Robust L1)": hgb_l1,
    }
    if LGBMRegressor is not None:
        models["LightGBM (Huber)"] = LGBMRegressor(
            objective="huber",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=min(31, max(3, n_train // 2)),
            min_child_samples=min_leaf_gbdt,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        models["Ensemble (RF + HistGBR + LightGBM)"] = EnsembleRegressor(
            [rf, hgb_l2, models["LightGBM (Huber)"]]
        )
    else:
        models["Ensemble (RF + HistGBR)"] = EnsembleRegressor(
            [rf, hgb_l2, hgb_l1], weights=[0.35, 0.40, 0.25]
        )

    return models


def _fit_with_optional_weights(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
) -> BaseEstimator:
    fitted = clone(model)
    if sample_weight is None:
        fitted.fit(X, y)
        return fitted

    if hasattr(fitted, "steps"):
        last_name = fitted.steps[-1][0]
        fitted.fit(X, y, **{f"{last_name}__sample_weight": sample_weight})
    else:
        fitted.fit(X, y, sample_weight=sample_weight)
    return fitted


def run_predict_cli(argv: Sequence[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="train_descriptors.py predict",
        description="Predict crystal solvent content (%) from Patterson + diffraction metadata.",
    )
    parser.add_argument("input_path", help="Path to .mtz, .cif, .map(.gz), .npy, or .txt list of files.")
    parser.add_argument("--model", type=Path, default=Path("descriptor_model.joblib"))
    parser.add_argument(
        "--d-min-hint",
        type=float,
        default=None,
        help="Optional native d_min (A) hint when input is a .map file lacking reflection limits.",
    )
    args = parser.parse_args(list(argv))

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feat_cols: List[str] = bundle["feature_names"]
    grid_size = tuple(bundle.get("grid_size", DEFAULT_GRID_SIZE))
    grid_spacing = float(bundle.get("grid_spacing", DEFAULT_GRID_SPACING))
    d_min = bundle.get("d_min", DEFAULT_D_MIN)
    origin_mask_radius = float(bundle.get("origin_mask_radius", DEFAULT_ORIGIN_MASK_RADIUS))

    in_path = Path(args.input_path)
    if in_path.suffix.lower() == ".txt":
        with open(in_path, "r", encoding="utf-8") as f:
            files = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        files = [str(in_path)]

    for fp in files:
        desc = extract_descriptors_for_file(
            file_path=fp,
            grid_size=grid_size,  # type: ignore[arg-type]
            grid_spacing=grid_spacing,
            d_min=d_min,
            origin_mask_radius=origin_mask_radius,
        )
        if args.d_min_hint is not None and not math.isfinite(desc.get("native_d_min", float("nan"))):
            desc["native_d_min"] = args.d_min_hint
        vec = np.array([[desc.get(col, float("nan")) for col in feat_cols]], dtype=np.float64)
        vec = np.where(np.isfinite(vec), vec, np.nan)
        pred = float(np.clip(model.predict(vec)[0], 0.0, 100.0))
        print(f"{Path(fp).name}: {round(pred, 3)}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        run_predict_cli(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(
        description="Train and evaluate RF & GBDT models on 3D Patterson + diffraction metadata."
    )
    parser.add_argument("training_file", type=Path, help="Training split file (name solvent_pct path).")
    parser.add_argument("validation_file", type=Path, help="Validation split file.")
    parser.add_argument(
        "--descriptors",
        type=Path,
        default=Path("additional_descriptors.txt"),
        help="Descriptor table produced by generate_descriptors.py.",
    )
    parser.add_argument(
        "--map-metadata",
        type=Path,
        default=None,
        help="Optional map_metadata.tsv from generate_maps.py to backfill diffraction metadata.",
    )
    parser.add_argument(
        "--feature-set",
        choices=("all", "patterson_only", "diffraction_only"),
        default="all",
        help="Feature subset to train on (default: all = 55 Patterson + 15 diffraction/cell features).",
    )
    parser.add_argument("--test-file", type=Path, default=None, help="Optional held-out test split file.")
    parser.add_argument(
        "--sample-weighting",
        choices=("inverse_density", "linear_tail", "none"),
        default="inverse_density",
        help="Training sample weighting scheme for low/high solvent tails (default: inverse_density).",
    )
    parser.add_argument("--loss-weight", type=float, default=8.0)
    parser.add_argument("--output-model", type=Path, default=Path("descriptor_model.joblib"))
    parser.add_argument("--predictions-out", type=Path, default=Path("descriptor_validation_predictions.txt"))
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE[0])
    parser.add_argument("--grid-spacing", type=float, default=DEFAULT_GRID_SPACING)
    parser.add_argument("--d-min", type=float, default=DEFAULT_D_MIN)
    parser.add_argument("--origin-mask-radius", type=float, default=DEFAULT_ORIGIN_MASK_RADIUS)
    parser.add_argument("--perm-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_feat_names, desc_map = load_descriptor_table(args.descriptors)
    all_feat_names, desc_map = backfill_from_map_metadata(all_feat_names, desc_map, args.map_metadata)

    _, X_train_full, y_train = align_split_features(load_split_targets(args.training_file), desc_map)
    val_names, X_val_full, y_val = align_split_features(load_split_targets(args.validation_file), desc_map)

    feat_names, keep_idx = select_feature_subset(all_feat_names, X_train_full, args.feature_set)
    X_train = X_train_full[:, keep_idx]
    X_val = X_val_full[:, keep_idx]

    print(
        f"Loaded {len(feat_names)} informative features (mode='{args.feature_set}') | "
        f"train={len(y_train)}, val={len(y_val)}"
    )

    sample_weights = compute_sample_weights(
        y_train, mode=args.sample_weighting, tail_weight=args.loss_weight
    )

    # Print feature-family ablation when both Patterson & informative diffraction metadata are present
    if args.feature_set == "all":
        diff_idx = [
            i for i in keep_idx if all_feat_names[i] in set(DIFFRACTION_METADATA_COLUMNS)
        ]
        patt_idx = [
            i for i in keep_idx if all_feat_names[i] in set(PATTERSON_ONLY_COLUMNS)
        ]
        if diff_idx and patt_idx:
            probe = build_candidate_models(len(y_train), seed=args.seed)["HistGBR (SquaredError)"]
            p_diff = np.clip(_fit_with_optional_weights(probe, X_train_full[:, diff_idx], y_train, sample_weights).predict(X_val_full[:, diff_idx]), 0.0, 100.0)
            p_patt = np.clip(_fit_with_optional_weights(probe, X_train_full[:, patt_idx], y_train, sample_weights).predict(X_val_full[:, patt_idx]), 0.0, 100.0)
            p_both = np.clip(_fit_with_optional_weights(probe, X_train, y_train, sample_weights).predict(X_val), 0.0, 100.0)
            s_diff = evaluate_predictions(y_val, p_diff, loss_weight=args.loss_weight)
            s_patt = evaluate_predictions(y_val, p_patt, loss_weight=args.loss_weight)
            s_both = evaluate_predictions(y_val, p_both, loss_weight=args.loss_weight)
            print(
                f"Feature-family ablation (HistGBR Val RMSE): "
                f"Diffraction/Cell Only={s_diff['rmse']:.2f}% | "
                f"Patterson Only={s_patt['rmse']:.2f}% | "
                f"Combined={s_both['rmse']:.2f}%"
            )

    candidates = build_candidate_models(len(y_train), seed=args.seed)

    best_name = ""
    best_model: Optional[BaseEstimator] = None
    best_val_rmse = float("inf")
    best_val_preds = np.zeros_like(y_val)
    best_metrics: Dict[str, float] = {}

    print(
        f"\n{'Model':<36} {'RMSE (%)':>9} {'MAE (%)':>9} {'Tail MAE':>9} "
        f"{'W-MSE':>9} {'Pearson r':>10} {'R^2':>8}"
    )
    print("-" * 96)

    for name, template in candidates.items():
        fitted = _fit_with_optional_weights(template, X_train, y_train, sample_weights)
        val_pred = np.clip(fitted.predict(X_val), 0.0, 100.0)
        stats = evaluate_predictions(y_val, val_pred, loss_weight=args.loss_weight)
        print(
            f"{name:<36} {stats['rmse']:>9.2f} {stats['mae']:>9.2f} {stats['tail_mae']:>9.2f} "
            f"{stats['wmse']:>9.5f} {stats['r']:>10.4f} {stats['r2']:>8.4f}"
        )
        if stats["rmse"] < best_val_rmse:
            best_val_rmse = stats["rmse"]
            best_name = name
            best_model = fitted
            best_val_preds = val_pred
            best_metrics = stats

    assert best_model is not None
    print(f"\nSelected best validation model: {best_name} (Val RMSE={best_val_rmse:.2f}%)")

    if len(y_val) >= 4 and args.perm_repeats > 0:
        perm = permutation_importance(
            best_model,
            X_val,
            y_val,
            scoring="neg_root_mean_squared_error",
            n_repeats=args.perm_repeats,
            random_state=args.seed,
        )
        top_idx = np.argsort(perm.importances_mean)[::-1][:15]
        print("\nTop descriptors (validation permutation importance, +RMSE %):")
        for rank, idx in enumerate(top_idx, start=1):
            print(
                f"  {rank:2d}. {feat_names[idx]:<28} "
                f"{perm.importances_mean[idx]:>+7.3f} +/- {perm.importances_std[idx]:.3f}"
            )

    if args.test_file is not None and args.test_file.is_file():
        _, X_test_full, y_test = align_split_features(load_split_targets(args.test_file), desc_map)
        X_test = X_test_full[:, keep_idx]
        test_pred = np.clip(best_model.predict(X_test), 0.0, 100.0)
        t_stats = evaluate_predictions(y_test, test_pred, loss_weight=args.loss_weight)
        print(
            f"\nHeld-out TEST ({best_name}, n={len(y_test)}): "
            f"RMSE={t_stats['rmse']:.2f}%, MAE={t_stats['mae']:.2f}%, "
            f"Tail MAE={t_stats['tail_mae']:.2f}%, r={t_stats['r']:.4f}, R^2={t_stats['r2']:.4f}"
        )

    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.predictions_out, "w", encoding="utf-8") as pf:
        for name, yt, yp in zip(val_names, y_val / 100.0, best_val_preds / 100.0):
            pf.write(f"{name} {yt:.6f} {yp:.6f}\n")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": best_name,
            "model": best_model,
            "feature_names": feat_names,
            "feature_set": args.feature_set,
            "grid_size": (args.grid_size, args.grid_size, args.grid_size),
            "grid_spacing": args.grid_spacing,
            "d_min": args.d_min if (args.d_min is not None and args.d_min > 0) else None,
            "origin_mask_radius": args.origin_mask_radius,
            "val_metrics": best_metrics,
        },
        args.output_model,
    )
    print(f"Saved trained model bundle to {args.output_model}")


if __name__ == "__main__":
    main()
