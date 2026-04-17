"""Tests for training configuration."""

import pytest

from src.config import TrainConfig, DEFECT_LABELS


class TestTrainConfig:
    def test_default_config(self):
        config = TrainConfig()
        assert config.num_classes == 5
        assert config.batch_size == 16
        assert config.img_size == 640
        assert config.epochs == 100

    def test_class_names_match_labels(self):
        config = TrainConfig()
        assert len(config.class_names) == config.num_classes
        assert len(DEFECT_LABELS) == config.num_classes

    def test_confidence_threshold_range(self):
        config = TrainConfig()
        assert 0.0 < config.conf_threshold < 1.0
        assert 0.0 < config.iou_threshold < 1.0

    def test_custom_config(self):
        config = TrainConfig(epochs=50, batch_size=8, img_size=416)
        assert config.epochs == 50
        assert config.batch_size == 8
        assert config.img_size == 416

    def test_augmentation_defaults(self):
        config = TrainConfig()
        assert config.augment is True
        assert 0.0 <= config.mosaic <= 1.0
        assert 0.0 <= config.mixup <= 1.0
        assert 0.0 <= config.flipud <= 1.0
        assert 0.0 <= config.fliplr <= 1.0
