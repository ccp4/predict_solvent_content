#!/usr/bin/env python3
"""
Cluster curated PDB entries at <30% sequence identity (plus crystal-form grouping),
create leak-free stratified train/val/test splits for 3D Patterson models, and
evaluate the two mandatory Priority 1 crystallographic baselines:

  - Baseline A (Metadata-only): Predict solvent % from (d_min, V_asu, V_cell,
    spacegroup_number, point_group_order) without looking at Patterson voxels,
    plus a d_min-only control quantifying the PDB resolution-solvent confounder.
  - Baseline B (Matthews probability): Analytical Kantardjieff & Rupp (2003) /
    Matthews prediction (both MAP copy-number and posterior expectation) when
    protomer sequence MW is known.
"""

import argparse
from collections import defaultdict
import csv
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import gemmi
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

from getsolv import matthews_probabilities


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent: Dict[str, str] = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb


def _read_fasta_by_pdb(fasta_path: Path) -> Dict[str, List[str]]:
    seqs_by_pdb: Dict[str, List[str]] = defaultdict(list)
    if not fasta_path.is_file():
        return seqs_by_pdb

    cur_pdb: Optional[str] = None
    cur_chunks: List[str] = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_pdb is not None and cur_chunks:
                    seqs_by_pdb[cur_pdb].append("".join(cur_chunks))
                header_id = line[1:].split()[0]
                cur_pdb = header_id.split("_")[0].lower()
                cur_chunks = []
            else:
                cur_chunks.append(line.upper())
        if cur_pdb is not None and cur_chunks:
            seqs_by_pdb[cur_pdb].append("".join(cur_chunks))
    return seqs_by_pdb


def cluster_by_sequence_identity(
    pdb_codes: Sequence[str],
    fasta_path: Path,
    min_seq_id: float = 0.30,
) -> Dict[str, str]:
    """
    Group PDB codes into connected components sharing any polymer chain with
    >= `min_seq_id` sequence identity.

    Uses `mmseqs easy-cluster` automatically if installed on PATH (for 100k+ PDB scale),
    otherwise uses a k-mer prefiltered C++ BLOSUM62 global/semi-global alignment via Gemmi.
    """
    uf = _UnionFind(pdb_codes)
    valid_set = set(pdb_codes)

    if not fasta_path.is_file() or fasta_path.stat().st_size == 0:
        return {code: uf.find(code) for code in pdb_codes}

    mmseqs_bin = shutil.which("mmseqs")
    if mmseqs_bin is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_prefix = os.path.join(tmpdir, "clu")
            tmp_work = os.path.join(tmpdir, "work")
            cmd = [
                mmseqs_bin,
                "easy-cluster",
                str(fasta_path),
                out_prefix,
                tmp_work,
                "--min-seq-id",
                f"{min_seq_id:.3f}",
                "-c",
                "0.8",
                "--cov-mode",
                "0",
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            tsv_file = f"{out_prefix}_cluster.tsv"
            with open(tsv_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        p1 = parts[0].split("_")[0].lower()
                        p2 = parts[1].split("_")[0].lower()
                        if p1 in valid_set and p2 in valid_set:
                            uf.union(p1, p2)
        return {code: uf.find(code) for code in pdb_codes}

    # Built-in Gemmi BLOSUM62 C++ alignment with k-mer candidate prefiltering
    seqs_by_pdb = _read_fasta_by_pdb(fasta_path)
    unique_seq_to_pdbs: Dict[str, List[str]] = defaultdict(list)
    for code in pdb_codes:
        for seq in seqs_by_pdb.get(code, ()):
            if len(seq) >= 10:
                unique_seq_to_pdbs[seq].append(code)

    # Identical sequences immediately union their PDB entries
    for pdbs in unique_seq_to_pdbs.values():
        first = pdbs[0]
        for other in pdbs[1:]:
            uf.union(first, other)

    unique_seqs = sorted(unique_seq_to_pdbs.keys(), key=len, reverse=True)
    # Build 3-mer inverted index (captures twilight-zone ~30% homologs via conserved motifs)
    k = 3
    kmer_index: Dict[str, List[int]] = defaultdict(list)
    seq_chars: List[List[str]] = [list(s) for s in unique_seqs]
    seq_kmer_sets: List[Set[str]] = []

    for idx, s in enumerate(unique_seqs):
        kmers = {s[pos : pos + k] for pos in range(max(0, len(s) - k + 1))}
        seq_kmer_sets.append(kmers)
        for km in kmers:
            kmer_index[km].append(idx)

    scoring = gemmi.AlignmentScoring("b")
    pct_threshold = min_seq_id * 100.0

    for i, s_i in enumerate(unique_seqs):
        rep_i = unique_seq_to_pdbs[s_i][0]
        len_i = len(s_i)
        cand_counts: Dict[int, int] = defaultdict(int)
        for km in seq_kmer_sets[i]:
            for j in kmer_index[km]:
                if j > i:
                    cand_counts[j] += 1

        for j, shared_kmers in cand_counts.items():
            s_j = unique_seqs[j]
            rep_j = unique_seq_to_pdbs[s_j][0]
            if uf.find(rep_i) == uf.find(rep_j):
                continue
            len_j = len(s_j)
            # Length ratio filter (at <50% length coverage of longer chain, avoid spurious motif hits)
            if len_j < 0.5 * len_i:
                continue
            # Require at least 2 shared 3-mers (or 1 for very short peptides) before full DP alignment
            min_shared = 1 if len_j < 35 else 2
            if shared_kmers < min_shared:
                continue

            aln = gemmi.align_string_sequences(seq_chars[i], seq_chars[j], [], scoring)
            # `calculate_identity(2)` normalizes by shorter sequence length
            if aln.calculate_identity(2) >= pct_threshold:
                uf.union(rep_i, rep_j)

    return {code: uf.find(code) for code in pdb_codes}


def stratified_cluster_split(
    records: List[Dict],
    cluster_map: Dict[str, str],
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Assign sequence + crystal-form clusters to 'train', 'val', or 'test' while
    stratifying across solvent-content quantiles and keeping whole clusters intact.
    """
    uf = _UnionFind([r["pdb_code"] for r in records])
    by_seq_cluster: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        c = r["pdb_code"]
        by_seq_cluster[cluster_map.get(c, c)].append(c)
    for members in by_seq_cluster.values():
        for m in members[1:]:
            uf.union(members[0], m)

    # Also group nearly identical crystal forms (same spacegroup + ASU volume within 1%)
    # if they have matching protomer MW within 2%
    sorted_by_asu = sorted(records, key=lambda x: (x["spacegroup_number"], x["asu_volume"]))
    for idx in range(1, len(sorted_by_asu)):
        prev_r = sorted_by_asu[idx - 1]
        curr_r = sorted_by_asu[idx]
        if prev_r["spacegroup_number"] == curr_r["spacegroup_number"]:
            v1, v2 = prev_r["asu_volume"], curr_r["asu_volume"]
            m1, m2 = prev_r["protomer_mw"], curr_r["protomer_mw"]
            if max(v1, v2) > 0 and abs(v1 - v2) / max(v1, v2) < 0.01:
                if max(m1, m2) > 0 and abs(m1 - m2) / max(m1, m2) < 0.02:
                    uf.union(prev_r["pdb_code"], curr_r["pdb_code"])

    clusters: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        clusters[uf.find(r["pdb_code"])].append(r)

    rng = np.random.default_rng(seed)
    cluster_items = list(clusters.items())
    rng.shuffle(cluster_items)
    # Sort clusters by mean solvent content to stratify across the full [15%, 90%] range
    cluster_items.sort(key=lambda kv: float(np.mean([x["real_solv"] for x in kv[1]])))

    n_total = len(records)
    target_val = max(1, round(n_total * val_frac)) if n_total >= 3 else 0
    target_test = max(1, round(n_total * test_frac)) if n_total >= 3 else 0

    assignments: Dict[str, str] = {}
    counts = {"train": 0, "val": 0, "test": 0}

    # Walk through solvent-sorted clusters in strata of ~10 clusters for balanced tails
    stratum_size = max(3, round(1.0 / max(0.05, min(val_frac, test_frac))))
    for start in range(0, len(cluster_items), stratum_size):
        block = cluster_items[start : start + stratum_size]
        rng.shuffle(block)
        for _, members in block:
            sz = len(members)
            # Compute deficit relative to desired cumulative proportions
            seen = max(1, sum(counts.values()) + sz)
            val_def = val_frac - (counts["val"] / seen)
            test_def = test_frac - (counts["test"] / seen)

            if counts["test"] < target_test and test_def >= val_def and test_def > 0:
                split = "test"
            elif counts["val"] < target_val and val_def > 0:
                split = "val"
            else:
                split = "train"

            counts[split] += sz
            for m in members:
                assignments[m["pdb_code"]] = split

    return assignments


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if np.sum(mask) == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "r": float("nan"), "r2": float("nan")}
    yt = y_true[mask]
    yp = y_pred[mask]
    rmse = float(math.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    if len(yt) >= 2 and np.std(yt) > 1e-12 and np.std(yp) > 1e-12:
        r = float(np.corrcoef(yt, yp)[0, 1])
        r2 = float(r2_score(yt, yp))
    else:
        r = float("nan")
        r2 = float("nan")
    return {"n": int(len(yt)), "rmse": rmse, "mae": mae, "r": r, "r2": r2}


def evaluate_baselines(
    records: List[Dict],
    splits: Dict[str, str],
    map_meta: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Train Baseline A (resolution-only and full crystallographic metadata ML models)
    on 'train' and evaluate alongside Baseline B (Matthews probability) on 'val' and 'test'.
    """
    by_split: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    for r in records:
        by_split[splits[r["pdb_code"]]].append(r)

    if len(by_split["train"]) < 2:
        raise ValueError("Need at least 2 training structures to fit metadata baselines.")

    def _feat_res_only(rows: List[Dict]) -> np.ndarray:
        return np.array([[r["resolution"]] for r in rows], dtype=np.float64)

    def _feat_meta(rows: List[Dict]) -> np.ndarray:
        feats = []
        for r in rows:
            row_f = [
                r["resolution"],
                math.log(max(1.0, r["asu_volume"])),
                math.log(max(1.0, r["cell_volume"])),
                float(r["spacegroup_number"]),
                float(r["point_group_order"]),
            ]
            if map_meta is not None:
                mm = map_meta.get(r["pdb_code"], {})
                row_f.extend([
                    mm.get("origin_to_sigma", float("nan")),
                    mm.get("low_res_intensity_ratio", float("nan")),
                    mm.get("acentric_e2_minus_1", float("nan")),
                ])
            feats.append(row_f)
        return np.asarray(feats, dtype=np.float64)

    y_train = np.array([r["real_solv"] for r in by_split["train"]], dtype=np.float64)
    train_mean = float(np.mean(y_train))

    res_only_model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(max_depth=3, random_state=42),
    )
    res_only_model.fit(_feat_res_only(by_split["train"]), y_train)

    meta_hgb_model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(max_depth=6, random_state=42),
    )
    meta_hgb_model.fit(_feat_meta(by_split["train"]), y_train)

    meta_rf_model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1),
    )
    meta_rf_model.fit(_feat_meta(by_split["train"]), y_train)

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for split_name in ("val", "test"):
        rows = by_split[split_name]
        if not rows:
            continue
        y_true = np.array([r["real_solv"] for r in rows], dtype=np.float64)

        null_pred = np.full_like(y_true, train_mean)
        res_pred = res_only_model.predict(_feat_res_only(rows))
        hgb_pred = meta_hgb_model.predict(_feat_meta(rows))
        rf_pred = meta_rf_model.predict(_feat_meta(rows))

        matt_map_pred = np.array([r["matthews_solv_map"] for r in rows], dtype=np.float64)
        matt_exp_pred = np.array([r["matthews_solv_expected"] for r in rows], dtype=np.float64)

        all_results[split_name] = {
            "Null (Train Mean)": _metrics_dict(y_true, null_pred),
            "Baseline A1: d_min Only (Confounder Check)": _metrics_dict(y_true, res_pred),
            "Baseline A2: Metadata HistGBR (d_min, V_asu, SG)": _metrics_dict(y_true, hgb_pred),
            "Baseline A3: Metadata RandomForest (d_min, V_asu, SG)": _metrics_dict(y_true, rf_pred),
            "Baseline B1: Matthews Probability (MAP N_mol, known MW)": _metrics_dict(y_true, matt_map_pred),
            "Baseline B2: Matthews Probability (Expected Vs, known MW)": _metrics_dict(y_true, matt_exp_pred),
        }

    return all_results


def _load_results_tsv(tsv_path: Path) -> List[Dict]:
    records: List[Dict] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records.append({
                "pdb_code": row["pdb_code"].strip().lower(),
                "real_solv": float(row["real_solv"]),
                "matthews_solv_map": float(row["matthews_solv_map"]),
                "matthews_solv_expected": float(row["matthews_solv_expected"]),
                "protomer_mw": float(row["protomer_mw"]),
                "cell_volume": float(row["cell_volume"]),
                "asu_volume": float(row["asu_volume"]),
                "spacegroup_number": int(row["spacegroup_number"]),
                "point_group_order": int(row["point_group_order"]),
                "resolution": float(row["resolution"]),
            })
    return records


def _load_map_metadata(meta_path: Optional[Path]) -> Tuple[Optional[Dict[str, Dict[str, float]]], Dict[str, str]]:
    if meta_path is None or not meta_path.is_file():
        return None, {}
    meta_dict: Dict[str, Dict[str, float]] = {}
    npy_paths: Dict[str, str] = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            code = row["name"].strip().lower()
            npy_paths[code] = row["npy_path"].strip()
            meta_dict[code] = {
                "origin_to_sigma": float(row["origin_to_sigma"]),
                "low_res_intensity_ratio": float(row["low_res_intensity_ratio"]),
                "acentric_e2_minus_1": float(row["acentric_e2_minus_1"]),
            }
    return meta_dict, npy_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create <30% sequence-identity clustered splits and run Priority 1 baselines."
    )
    parser.add_argument("--results", type=Path, default=Path("results.tsv"))
    parser.add_argument("--fasta", type=Path, default=Path("curated_sequences.fasta"))
    parser.add_argument("--map-metadata", type=Path, default=None, help="Optional map_metadata.tsv from generate_maps.py.")
    parser.add_argument("--map-dir", type=Path, default=None, help="Optional directory containing <pdb_code>.npy files.")
    parser.add_argument("--min-seq-id", type=float, default=0.30, help="Sequence identity clustering cutoff (default: 0.30).")
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    records = _load_results_tsv(args.results)
    map_meta, npy_by_code = _load_map_metadata(args.map_metadata)

    if args.map_dir is not None and args.map_dir.is_dir():
        for r in records:
            c = r["pdb_code"]
            cand = args.map_dir / f"{c}.npy"
            if cand.is_file():
                npy_by_code[c] = str(cand.resolve())

    pdb_codes = [r["pdb_code"] for r in records]
    print(f"Clustering {len(pdb_codes)} curated structures at <{args.min_seq_id * 100:.0f}% sequence identity...")
    cluster_map = cluster_by_sequence_identity(pdb_codes, args.fasta, min_seq_id=args.min_seq_id)
    n_clusters = len(set(cluster_map.values()))
    print(f"Formed {n_clusters} non-homologous sequence clusters from {len(pdb_codes)} entries.")

    splits = stratified_cluster_split(
        records,
        cluster_map=cluster_map,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_filenames = {
        "train": args.out_dir / "training_data.txt",
        "val": args.out_dir / "validation_data.txt",
        "test": args.out_dir / "test_data.txt",
    }
    handles = {k: open(p, "w", encoding="utf-8") for k, p in split_filenames.items()}
    try:
        with open(args.out_dir / "splits_summary.tsv", "w", encoding="utf-8") as sum_f:
            sum_f.write("pdb_code\tsplit\tseq_cluster\treal_solv\tresolution\tasu_volume\tnpy_path\n")
            for r in records:
                c = r["pdb_code"]
                sp = splits[c]
                npy_path = npy_by_code.get(c, f"{c}.npy")
                handles[sp].write(f"{c} {r['real_solv']:.3f} {npy_path}\n")
                sum_f.write(
                    f"{c}\t{sp}\t{cluster_map[c]}\t{r['real_solv']:.3f}\t"
                    f"{r['resolution']:.2f}\t{r['asu_volume']:.2f}\t{npy_path}\n"
                )
    finally:
        for h in handles.values():
            h.close()

    counts = {sp: sum(1 for r in records if splits[r["pdb_code"]] == sp) for sp in ("train", "val", "test")}
    print(f"Split sizes: train={counts['train']}, val={counts['val']}, test={counts['test']}")

    if counts["train"] >= 2 and (counts["val"] > 0 or counts["test"] > 0):
        bench = evaluate_baselines(records, splits, map_meta=map_meta)
        for sp_name, models in bench.items():
            print(f"\n=== Mandatory Baselines ({sp_name.upper()} set, n={counts[sp_name]}) ===")
            print(f"{'Model':<58} {'RMSE (%)':>9} {'MAE (%)':>9} {'Pearson r':>10} {'R^2':>8}")
            print("-" * 98)
            for m_name, m_stats in models.items():
                print(
                    f"{m_name:<58} {m_stats['rmse']:>9.2f} {m_stats['mae']:>9.2f} "
                    f"{m_stats['r']:>10.4f} {m_stats['r2']:>8.4f}"
                )


if __name__ == "__main__":
    main()
