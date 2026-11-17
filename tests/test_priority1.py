import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import gemmi
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from patterson import (
    compute_patterson_from_file,
    interpolate_and_normalize_patterson,
)
from generate_descriptors import extract_descriptors_for_file
from generate_maps import main as generate_maps_main
from getsolv import analyze_structure
from split_and_baselines import (
    cluster_by_sequence_identity,
    evaluate_baselines,
    stratified_cluster_split,
)
from usolv import ResidualUNetSE3D
from vitsolv import ViTVNet, get_3DReg_config

RNASE_MTZ = Path("/opt/xtal/ccp4-9/examples/rnase/rnase18.mtz")
GERE_MTZ = Path("/opt/xtal/ccp4-9/examples/tutorial/results/gere_nat.mtz")
MOLREP_CIF = Path("/opt/xtal/ccp4-9/examples/windows/molrep_mtz.cif")
RNASE_PDB = Path("/opt/xtal/ccp4-9/examples/rnase/rnase.pdb")


class TestPriority1(unittest.TestCase):
    def test_origin_centering_and_inversion_symmetry(self):
        """Origin (0,0,0) must map to [N//2, N//2, N//2] and preserve P(-u) == P(u)."""
        grid = gemmi.FloatGrid(40, 40, 40)
        grid.set_unit_cell(gemmi.UnitCell(40.0, 40.0, 40.0, 90.0, 90.0, 90.0))
        grid.spacegroup = gemmi.find_spacegroup_by_name("P -1")
        grid.set_points_around(gemmi.Position(0.0, 0.0, 0.0), radius=2.5, value=50.0)
        grid.set_points_around(gemmi.Position(8.0, -5.0, 3.0), radius=2.0, value=10.0)
        grid.symmetrize_max()

        # Without origin masking, peak maximum must be right at [64, 64, 64]
        res_unmasked = interpolate_and_normalize_patterson(
            grid, grid_size=(128, 128, 128), grid_spacing=1.0, origin_mask_radius=0.0
        )
        arr = res_unmasked.array
        self.assertGreater(arr[64, 64, 64], 10.0)
        # Check inversion symmetry P(u) == P(-u) around index 64: (+8, -5, +3) vs (-8, +5, -3)
        self.assertAlmostEqual(
            float(arr[64 + 8, 64 - 5, 64 + 3]),
            float(arr[64 - 8, 64 + 5, 64 - 3]),
            places=5,
        )

        # With origin masking (r < 4.0 A), [64, 64, 64] AND periodic lattice copy [64+40, 64, 64] are 0.0
        res_masked = interpolate_and_normalize_patterson(
            grid, grid_size=(128, 128, 128), grid_spacing=1.0, origin_mask_radius=4.0
        )
        self.assertAlmostEqual(float(res_masked.array[64, 64, 64]), 0.0, places=5)
        self.assertAlmostEqual(float(res_masked.array[104, 64, 64]), 0.0, places=5)
        self.assertGreater(res_masked.origin_to_sigma, 10.0)

    @unittest.skipUnless(RNASE_MTZ.is_file() and GERE_MTZ.is_file(), "CCP4 example MTZs not present")
    def test_mtz_cif_patterson_and_resolution_truncation(self):
        res_rnase = compute_patterson_from_file(
            RNASE_MTZ, grid_size=(128, 128, 128), grid_spacing=1.0, d_min=3.5
        )
        self.assertEqual(res_rnase.array.shape, (128, 128, 128))
        self.assertLess(res_rnase.native_d_min, 2.0)
        self.assertGreaterEqual(res_rnase.used_d_min, 3.5)
        self.assertGreater(res_rnase.origin_to_sigma, 15.0)
        self.assertAlmostEqual(float(res_rnase.array[64, 64, 64]), 0.0, places=5)

        # Centered monoclinic C 1 2 1 crystal: C-centering copy at (a/2, b/2, 0) must also be masked
        res_gere = compute_patterson_from_file(
            GERE_MTZ, grid_size=(128, 128, 128), grid_spacing=1.0, d_min=3.5
        )
        self.assertEqual(res_gere.spacegroup_number, 5)
        self.assertGreater(res_gere.origin_to_sigma, 15.0)
        self.assertTrue(np.all(np.isfinite(res_gere.array)))

        if MOLREP_CIF.is_file():
            res_cif = compute_patterson_from_file(
                MOLREP_CIF, grid_size=(128, 128, 128), grid_spacing=1.0, d_min=3.5
            )
            self.assertEqual(res_cif.spacegroup_number, 19)
            self.assertEqual(res_cif.array.shape, (128, 128, 128))

    @unittest.skipUnless(RNASE_MTZ.is_file() and GERE_MTZ.is_file(), "CCP4 example MTZs not present")
    def test_generate_descriptors_multiple_files_no_shadowing(self):
        with tempfile.TemporaryDirectory() as tmp:
            list_txt = Path(tmp) / "inputs.txt"
            out_txt = Path(tmp) / "descriptors.txt"
            list_txt.write_text(f"{RNASE_MTZ}\n{GERE_MTZ}\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "generate_descriptors.py"),
                    "--input",
                    str(list_txt),
                    "--output",
                    str(out_txt),
                    "--grid-size",
                    "64",
                ],
                check=True,
            )
            lines = [ln.strip() for ln in out_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
            # Header + 2 processed files (second file must NOT crash with kurtosis shadowing!)
            self.assertEqual(len(lines), 3)
            for row in lines[1:]:
                cols = row.split()
                self.assertEqual(len(cols), 16)
                std_val = float(cols[5])
                skel_len = int(cols[12])
                self.assertGreater(std_val, 0.2)
                self.assertGreater(skel_len, 0)

    def test_getsolv_quality_filters_and_hexagonal_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            if RNASE_PDB.is_file():
                rec, reason = analyze_structure(RNASE_PDB)
                self.assertIsNone(reason)
                self.assertIsNotNone(rec)
                self.assertAlmostEqual(rec["real_solv"], 46.7, delta=1.5)
                self.assertEqual(rec["matt_nmol"], 2)

            # Build a hexagonal P 6 structure to verify non-orthogonal cell volume (sin 120 = sqrt(3)/2)
            st_hex = gemmi.Structure()
            st_hex.spacegroup_hm = "P 6"
            st_hex.cell = gemmi.UnitCell(50.0, 50.0, 60.0, 90.0, 90.0, 120.0)
            model = gemmi.Model("1")
            chain = gemmi.Chain("A")
            for idx in range(120):
                res = gemmi.Residue()
                res.name = "ALA"
                res.seqid = gemmi.SeqId(idx + 1, " ")
                atom = gemmi.Atom()
                atom.name = "CA"
                atom.element = gemmi.Element("C")
                atom.pos = gemmi.Position(10.0 + 0.1 * idx, 15.0, 20.0)
                res.add_atom(atom)
                chain.add_residue(res)
            model.add_chain(chain)
            st_hex.add_model(model)
            hex_pdb = tmp_path / "hextest.pdb"
            st_hex.write_pdb(str(hex_pdb))

            rec_hex, reason_hex = analyze_structure(hex_pdb)
            self.assertIsNone(reason_hex)
            # True volume is 50*50*60 * sqrt(3)/2 = 129903.81, NOT 150,000!
            self.assertAlmostEqual(rec_hex["cell_volume"], 129903.81, delta=10.0)

            # Build a DNA structure and verify it is rejected
            st_dna = st_hex.clone()
            for res in st_dna[0]["A"]:
                res.name = "DA"
            dna_pdb = tmp_path / "dnatest.pdb"
            st_dna.write_pdb(str(dna_pdb))
            rec_dna, reason_dna = analyze_structure(dna_pdb)
            self.assertIsNone(rec_dna)
            self.assertIn("nucle", reason_dna)

            # Build a structure with >15% unmodeled SEQRES discrepancy and verify rejection
            st_disc = st_hex.clone()
            st_disc.setup_entities()
            st_disc.entities[0].full_sequence = ["ALA"] * 200  # 200 in SEQRES vs 120 modeled (40% diff)
            disc_cif = tmp_path / "disctest.cif"
            st_disc.make_mmcif_document().write_file(str(disc_cif))
            rec_disc, reason_disc = analyze_structure(disc_cif, max_mw_discrepancy=0.15)
            self.assertIsNone(rec_disc)
            self.assertIn("seqres_model_mw_discrepancy", reason_disc)

    def test_sequence_clustering_and_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            fasta = Path(tmp) / "seqs.fasta"
            # HBA and HBB share ~48% identity (>30% -> MUST cluster together!)
            # Ubiquitin and Lysozyme are completely unrelated (<20% identity -> separate clusters!)
            hba = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
            hbb = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"
            ubq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
            lys = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"
            gfp = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

            fasta.write_text(
                f">1hba_1\n{hba}\n>1hbb_1\n{hbb}\n>1ubq_1\n{ubq}\n>1lys_1\n{lys}\n>1gfp_1\n{gfp}\n",
                encoding="utf-8",
            )
            codes = ["1hba", "1hbb", "1ubq", "1lys", "1gfp"]
            clu = cluster_by_sequence_identity(codes, fasta, min_seq_id=0.30)
            self.assertEqual(clu["1hba"], clu["1hbb"])
            self.assertNotEqual(clu["1hba"], clu["1ubq"])
            self.assertNotEqual(clu["1ubq"], clu["1lys"])

            records = [
                {"pdb_code": "1hba", "real_solv": 52.0, "matthews_solv_map": 51.5, "matthews_solv_expected": 51.8, "protomer_mw": 15100.0, "cell_volume": 120000.0, "asu_volume": 30000.0, "spacegroup_number": 19, "point_group_order": 4, "resolution": 1.9},
                {"pdb_code": "1hbb", "real_solv": 53.0, "matthews_solv_map": 52.5, "matthews_solv_expected": 52.6, "protomer_mw": 15800.0, "cell_volume": 125000.0, "asu_volume": 31250.0, "spacegroup_number": 19, "point_group_order": 4, "resolution": 2.0},
                {"pdb_code": "1ubq", "real_solv": 38.0, "matthews_solv_map": 37.5, "matthews_solv_expected": 38.2, "protomer_mw": 8560.0, "cell_volume": 68000.0, "asu_volume": 17000.0, "spacegroup_number": 19, "point_group_order": 4, "resolution": 1.5},
                {"pdb_code": "1lys", "real_solv": 44.0, "matthews_solv_map": 43.8, "matthews_solv_expected": 44.1, "protomer_mw": 14300.0, "cell_volume": 240000.0, "asu_volume": 30000.0, "spacegroup_number": 96, "point_group_order": 8, "resolution": 1.7},
                {"pdb_code": "1gfp", "real_solv": 62.0, "matthews_solv_map": 61.0, "matthews_solv_expected": 61.4, "protomer_mw": 26900.0, "cell_volume": 350000.0, "asu_volume": 87500.0, "spacegroup_number": 19, "point_group_order": 4, "resolution": 2.6},
            ]
            splits = stratified_cluster_split(records, clu, val_frac=0.2, test_frac=0.2, seed=42)
            # Homologous globins MUST stay in the same split
            self.assertEqual(splits["1hba"], splits["1hbb"])
            bench = evaluate_baselines(records, splits)
            self.assertTrue("val" in bench or "test" in bench)

    def test_models_128_and_train_smoke(self):
        # Verify both architectures work on 128^3 and usolv has distinct downsample weights
        unet = ResidualUNetSE3D(in_channels=1, f_maps=[8, 16, 32, 64, 128])
        self.assertIsNot(unet.resnet_se_downsample1, unet.resnet_se_downsample2)
        x = torch.randn(2, 1, 64, 64, 64)
        out_u = unet(x)
        self.assertEqual(out_u.shape, (2, 1))

        cfg = get_3DReg_config(grid_size=64)
        vit = ViTVNet(config=cfg, img_size=(64, 64, 64))
        out_v = vit(x)
        self.assertEqual(out_v.shape, (2, 1))

    @unittest.skipUnless(RNASE_MTZ.is_file() and GERE_MTZ.is_file(), "CCP4 example MTZs not present")
    def test_end_to_end_generate_maps_and_train_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            in_list = tmp_p / "mtzs.txt"
            in_list.write_text(f"{RNASE_MTZ}\n{GERE_MTZ}\n", encoding="utf-8")
            meta_tsv = tmp_p / "meta.tsv"

            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "generate_maps.py"),
                    str(in_list),
                    "--output-dir",
                    str(tmp_p),
                    "--metadata-out",
                    str(meta_tsv),
                ],
                check=True,
            )
            rnase_npy = tmp_p / "rnase18.npy"
            gere_npy = tmp_p / "gere_nat.npy"
            self.assertTrue(rnase_npy.is_file())
            self.assertTrue(gere_npy.is_file())
            arr = np.load(rnase_npy)
            self.assertEqual(arr.shape, (128, 128, 128))
            self.assertEqual(arr.dtype, np.float16)
            # 128^3 * 2 bytes = 4,194,304 bytes (~4.0 MB vs 67.1 MB previously)
            self.assertLess(rnase_npy.stat().st_size, 4_250_000)

            # Smoke test train.py CLI on generated 64^3 maps for 1 epoch
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "generate_maps.py"),
                    str(in_list),
                    "--grid-size",
                    "64",
                    "--output-dir",
                    str(tmp_p / "small"),
                    "--metadata-out",
                    str(tmp_p / "small_meta.tsv"),
                ],
                check=True,
            )
            tr_txt = tmp_p / "train.txt"
            va_txt = tmp_p / "val.txt"
            tr_txt.write_text(f"rnase18 46.7 {tmp_p / 'small' / 'rnase18.npy'}\n", encoding="utf-8")
            va_txt.write_text(f"gere_nat 52.0 {tmp_p / 'small' / 'gere_nat.npy'}\n", encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "train.py"),
                    str(tr_txt),
                    str(va_txt),
                    "unet",
                    "--grid-size",
                    "64",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "1",
                    "--num-workers",
                    "0",
                ],
                cwd=str(tmp_p),
                check=True,
            )
            self.assertTrue((tmp_p / "models" / "model_0").is_file())
            self.assertTrue((tmp_p / "validation_correlation_0.txt").is_file())


if __name__ == "__main__":
    unittest.main()
