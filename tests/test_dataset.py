"""Tests for dataset preparation and validation."""

from pathlib import Path

from src.dataset import create_synthetic_dataset, validate_dataset


class TestSyntheticDataset:
    def test_create_synthetic(self, tmp_path: Path):
        create_synthetic_dataset(tmp_path, n_images=30, seed=42)

        for split in ["train", "val", "test"]:
            img_dir = tmp_path / "images" / split
            lbl_dir = tmp_path / "labels" / split
            assert img_dir.exists()
            assert lbl_dir.exists()
            assert len(list(img_dir.glob("*.jpg"))) > 0
            assert len(list(lbl_dir.glob("*.txt"))) > 0

    def test_label_format(self, tmp_path: Path):
        create_synthetic_dataset(tmp_path, n_images=10, seed=42)

        label_files = list((tmp_path / "labels" / "train").glob("*.txt"))
        assert len(label_files) > 0

        for lf in label_files:
            with open(lf) as f:
                for line in f:
                    parts = line.strip().split()
                    assert len(parts) == 5
                    cls = int(parts[0])
                    assert 0 <= cls <= 4
                    for val in parts[1:]:
                        v = float(val)
                        assert 0.0 <= v <= 1.0

    def test_image_label_pairing(self, tmp_path: Path):
        create_synthetic_dataset(tmp_path, n_images=20, seed=42)

        for split in ["train", "val", "test"]:
            images = {p.stem for p in (tmp_path / "images" / split).glob("*.jpg")}
            labels = {p.stem for p in (tmp_path / "labels" / split).glob("*.txt")}
            assert images == labels

    def test_reproducible_with_seed(self, tmp_path: Path):
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        create_synthetic_dataset(dir1, n_images=10, seed=42)
        create_synthetic_dataset(dir2, n_images=10, seed=42)

        labels1 = sorted((dir1 / "labels" / "train").glob("*.txt"))
        labels2 = sorted((dir2 / "labels" / "train").glob("*.txt"))
        assert len(labels1) == len(labels2)

        for l1, l2 in zip(labels1, labels2):
            assert l1.read_text() == l2.read_text()


class TestValidateDataset:
    def test_validate_synthetic(self, tmp_path: Path):
        create_synthetic_dataset(tmp_path, n_images=30, seed=42)

        yaml_path = tmp_path / "weld_defect.yaml"
        yaml_path.write_text(
            f"path: {tmp_path}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            f"nc: 5\nnames:\n  0: crack\n  1: porosity\n  2: spatter\n  3: undercut\n  4: overlap\n"
        )

        stats = validate_dataset(yaml_path)
        assert stats["total_images"] > 0
        assert stats["total_labels"] > 0

    def test_validate_missing_dir(self, tmp_path: Path):
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text(
            "path: nonexistent\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 5\nnames:\n  0: a\n"
        )
        stats = validate_dataset(yaml_path)
        assert stats["total_images"] == 0
