"""
Unit tests for hparam_paths module.
"""

import unittest
import tempfile
from pathlib import Path
from pydantic import BaseModel, Field
from hparam_paths import (
    parse_tyro_overrides,
    get_short_name_for_field,
    build_hparam_path,
    create_run_directory,
)


# Test fixtures - mock Pydantic models
class NestedConfig(BaseModel):
    learning_rate: float = Field(1e-3, json_schema_extra={"short_name": "lr"})
    batch_size: int = Field(32, json_schema_extra={"short_name": "bs"})
    epochs: int = Field(10, json_schema_extra={"short_name": "ep"})
    optimizer: str = "Adam"  # No short name
    use_tf32_matmul: bool = Field(False, json_schema_extra={"short_name": "tf32"})


class TestModel(BaseModel):
    latent_dims: int = Field(64, json_schema_extra={"short_name": "ld"})
    hidden_units: int = 128  # No short name
    use_batch_norm: bool = True  # Boolean without short name
    training: NestedConfig


class TestParseTypoOverrides(unittest.TestCase):
    """Tests for parse_tyro_overrides function."""

    def test_simple_overrides(self):
        args = ['--learning-rate', '0.001', '--batch-size', '64']
        result = parse_tyro_overrides(args)
        self.assertEqual(result, [('learning_rate', '0.001'), ('batch_size', '64')])

    def test_nested_overrides(self):
        args = ['--training.learning-rate', '0.001', '--latent-dims', '128']
        result = parse_tyro_overrides(args)
        self.assertEqual(result, [('training.learning_rate', '0.001'), ('latent_dims', '128')])

    def test_mixed_args_with_flags(self):
        args = ['--learning-rate', '0.001', '--verbose', '--batch-size', '32']
        result = parse_tyro_overrides(args)
        # Flags without values are now treated as boolean True
        self.assertEqual(result, [('learning_rate', '0.001'), ('verbose', 'True'), ('batch_size', '32')])

    def test_empty_args(self):
        args = []
        result = parse_tyro_overrides(args)
        self.assertEqual(result, [])

    def test_no_overrides(self):
        args = ['--help']
        result = parse_tyro_overrides(args)
        # --help is now treated as a boolean flag
        self.assertEqual(result, [('help', 'True')])

    def test_preserves_order(self):
        args = ['--z', '1', '--a', '2', '--m', '3']
        result = parse_tyro_overrides(args)
        self.assertEqual(result, [('z', '1'), ('a', '2'), ('m', '3')])

    def test_boolean_flag_true(self):
        """Test that boolean flags (--param) are captured as 'True'."""
        args = ['--use-batch-norm', '--latent-dims', '128']
        result = parse_tyro_overrides(args)
        # This should capture the boolean flag
        self.assertEqual(result, [('use_batch_norm', 'True'), ('latent_dims', '128')])

    def test_boolean_flag_false(self):
        """Test that negated boolean flags (--no-param) are captured as 'False'."""
        args = ['--no-use-batch-norm', '--latent-dims', '128']
        result = parse_tyro_overrides(args)
        # This should capture the negated boolean flag
        self.assertEqual(result, [('use_batch_norm', 'False'), ('latent_dims', '128')])

    def test_boolean_nested_flag_true(self):
        """Test that nested boolean flags work correctly."""
        args = ['--training.use-tf32-matmul', '--training.learning-rate', '0.001']
        result = parse_tyro_overrides(args)
        self.assertEqual(result, [('training.use_tf32_matmul', 'True'), ('training.learning_rate', '0.001')])

    def test_boolean_nested_flag_false(self):
        """Test that nested negated boolean flags work correctly."""
        args = ['--no-training.use-tf32-matmul', '--training.learning-rate', '0.001']
        result = parse_tyro_overrides(args)
        self.assertEqual(result, [('training.use_tf32_matmul', 'False'), ('training.learning_rate', '0.001')])

    def test_boolean_explicit_value(self):
        """Test that boolean flags with explicit values still work."""
        args = ['--use-batch-norm', 'True', '--latent-dims', '128']
        result = parse_tyro_overrides(args)
        self.assertEqual(result, [('use_batch_norm', 'True'), ('latent_dims', '128')])


class TestGetShortNameForField(unittest.TestCase):
    """Tests for get_short_name_for_field function."""

    def test_top_level_field_with_short_name(self):
        result = get_short_name_for_field(TestModel, 'latent_dims')
        self.assertEqual(result, 'ld')

    def test_top_level_field_without_short_name(self):
        result = get_short_name_for_field(TestModel, 'hidden_units')
        self.assertIsNone(result)

    def test_nested_field_with_short_name(self):
        result = get_short_name_for_field(TestModel, 'training.learning_rate')
        self.assertEqual(result, 'lr')

    def test_nested_field_without_short_name(self):
        result = get_short_name_for_field(TestModel, 'training.optimizer')
        self.assertIsNone(result)

    def test_nonexistent_field(self):
        result = get_short_name_for_field(TestModel, 'nonexistent')
        self.assertIsNone(result)

    def test_nonexistent_nested_field(self):
        result = get_short_name_for_field(TestModel, 'training.nonexistent')
        self.assertIsNone(result)


class TestBuildHparamPath(unittest.TestCase):
    """Tests for build_hparam_path function."""

    def test_single_override_with_short_name(self):
        args = ['--latent-dims', '128']
        result = build_hparam_path(args, TestModel)
        self.assertEqual(result, Path('ld128'))

    def test_multiple_overrides_with_short_names(self):
        args = ['--training.learning-rate', '0.001', '--training.batch-size', '64']
        result = build_hparam_path(args, TestModel)
        self.assertEqual(result, Path('lr0.001/bs64'))

    def test_override_without_short_name(self):
        args = ['--hidden-units', '256']
        result = build_hparam_path(args, TestModel)
        self.assertEqual(result, Path('hidden_units256'))

    def test_mixed_overrides(self):
        args = ['--latent-dims', '128', '--hidden-units', '256', '--training.learning-rate', '0.001']
        result = build_hparam_path(args, TestModel)
        self.assertEqual(result, Path('ld128/hidden_units256/lr0.001'))

    def test_no_overrides(self):
        args = []
        result = build_hparam_path(args, TestModel)
        self.assertEqual(result, Path('.'))

    def test_preserves_override_order(self):
        args = ['--training.batch-size', '64', '--latent-dims', '128', '--training.learning-rate', '0.001']
        result = build_hparam_path(args, TestModel)
        self.assertEqual(result, Path('bs64/ld128/lr0.001'))

    def test_boolean_flag_in_path_with_short_name(self):
        """Test that boolean flags with short names appear in directory path."""
        args = ['--training.use-tf32-matmul', '--training.learning-rate', '0.001']
        result = build_hparam_path(args, TestModel)
        self.assertEqual(result, Path('tf32True/lr0.001'))

    def test_boolean_flag_in_path_without_short_name(self):
        """Test that boolean flags without short names appear in directory path."""
        args = ['--use-batch-norm', '--latent-dims', '128']
        result = build_hparam_path(args, TestModel)
        self.assertEqual(result, Path('use_batch_normTrue/ld128'))

    def test_boolean_negated_flag_in_path(self):
        """Test that negated boolean flags appear correctly in directory path."""
        args = ['--no-training.use-tf32-matmul', '--training.learning-rate', '0.001']
        result = build_hparam_path(args, TestModel)
        self.assertEqual(result, Path('tf32False/lr0.001'))


class TestCreateRunDirectory(unittest.TestCase):
    """Tests for create_run_directory function."""

    def test_directory_structure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = ['--latent-dims', '128', '--training.learning-rate', '0.001']
            result = create_run_directory(
                expt_code='test_exp',
                tyro_args=args,
                model_class=TestModel,
                commit_hash='abc123',
                base_dir=tmp_path,
            )

            # Check that directory exists
            self.assertTrue(result.exists())
            self.assertTrue(result.is_dir())

            # Check structure: base_dir/expt_date_hash/ld128/lr0.001/uuid/
            self.assertEqual(result.parent.parent.parent.parent, tmp_path)
            self.assertIn('test_exp_', result.parent.parent.parent.name)
            self.assertIn('_abc123', result.parent.parent.parent.name)
            self.assertEqual(result.parent.parent.name, 'ld128')
            self.assertEqual(result.parent.name, 'lr0.001')
            # UUID should be 6 characters
            self.assertEqual(len(result.name), 6)

    def test_no_overrides_structure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = []
            result = create_run_directory(
                expt_code='test_exp',
                tyro_args=args,
                model_class=TestModel,
                commit_hash='abc123',
                base_dir=tmp_path,
            )

            self.assertTrue(result.exists())
            # With no overrides, should be: base_dir/expt_date_hash/./uuid/
            # The '.' gets normalized away, so we get: base_dir/expt_date_hash/uuid/
            self.assertEqual(result.parent.parent, tmp_path)
            self.assertIn('test_exp_', result.parent.name)
            self.assertIn('_abc123', result.parent.name)

    def test_creates_nested_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = ['--latent-dims', '128', '--training.batch-size', '64', '--training.learning-rate', '0.001']
            result = create_run_directory(
                expt_code='test_exp',
                tyro_args=args,
                model_class=TestModel,
                commit_hash='abc123',
                base_dir=tmp_path,
            )

            self.assertTrue(result.exists())
            # Should create: base_dir/expt_date_hash/ld128/bs64/lr0.001/uuid/
            self.assertEqual(result.parent.name, 'lr0.001')
            self.assertEqual(result.parent.parent.name, 'bs64')
            self.assertEqual(result.parent.parent.parent.name, 'ld128')

    def test_unique_uuids(self):
        """Test that multiple calls create unique directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = ['--latent-dims', '128']
            result1 = create_run_directory(
                expt_code='test_exp',
                tyro_args=args,
                model_class=TestModel,
                commit_hash='abc123',
                base_dir=tmp_path,
            )
            result2 = create_run_directory(
                expt_code='test_exp',
                tyro_args=args,
                model_class=TestModel,
                commit_hash='abc123',
                base_dir=tmp_path,
            )

            self.assertNotEqual(result1, result2)
            self.assertTrue(result1.exists())
            self.assertTrue(result2.exists())
            # Should have same parent (same hyperparameters)
            self.assertEqual(result1.parent, result2.parent)

    def test_symlink_created_and_not_broken(self):
        """Test that 'latest' symlink is created and points to the correct directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = ['--latent-dims', '128', '--training.learning-rate', '0.001']
            result = create_run_directory(
                expt_code='test_exp',
                tyro_args=args,
                model_class=TestModel,
                commit_hash='abc123',
                base_dir=tmp_path,
            )

            # Find the experiment directory (test_exp_<date>_abc123)
            expt_dir = result.parent.parent.parent
            symlink_path = expt_dir / "latest"

            # Verify symlink exists
            self.assertTrue(symlink_path.exists(), f"Symlink does not exist at {symlink_path}")
            self.assertTrue(symlink_path.is_symlink(), f"Path {symlink_path} is not a symlink")

            # Verify symlink is not broken (points to an existing directory)
            self.assertTrue(symlink_path.resolve().exists(), "Symlink is broken - target does not exist")
            self.assertTrue(symlink_path.resolve().is_dir(), "Symlink target is not a directory")

            # Verify symlink points to the created run directory
            self.assertEqual(symlink_path.resolve(), result.resolve(),
                           f"Symlink points to {symlink_path.resolve()} but expected {result.resolve()}")

            # Verify we can access files through the symlink by creating a test file
            test_file = result / "test.txt"
            test_file.write_text("test content")
            symlink_test_file = symlink_path / "test.txt"
            self.assertTrue(symlink_test_file.exists(), "Cannot access files through symlink")
            self.assertEqual(symlink_test_file.read_text(), "test content",
                           "File content accessed through symlink is incorrect")

    def test_symlink_updates_on_new_run(self):
        """Test that 'latest' symlink updates to point to the newest run directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            args = ['--latent-dims', '128']

            # Create first run
            result1 = create_run_directory(
                expt_code='test_exp',
                tyro_args=args,
                model_class=TestModel,
                commit_hash='abc123',
                base_dir=tmp_path,
            )

            expt_dir = result1.parent.parent
            symlink_path = expt_dir / "latest"

            # Verify symlink points to first run
            self.assertEqual(symlink_path.resolve(), result1.resolve())

            # Create second run (should update symlink)
            result2 = create_run_directory(
                expt_code='test_exp',
                tyro_args=args,
                model_class=TestModel,
                commit_hash='abc123',
                base_dir=tmp_path,
            )

            # Verify symlink now points to second run
            self.assertEqual(symlink_path.resolve(), result2.resolve(),
                           "Symlink should update to point to the newest run")
            self.assertNotEqual(symlink_path.resolve(), result1.resolve(),
                              "Symlink should no longer point to the first run")


if __name__ == "__main__":
    unittest.main()
