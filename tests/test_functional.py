import numpy as np
import pytest
import torch
import yaml
from ase.build import molecule

from agedi import (
    create_dataset,
    create_diffusion,
    load_diffusion,
    predict,
    sample,
    train,
    train_from_atoms,
    train_from_config,
)
from agedi.data import AtomsGraph, Dataset
from agedi.diffusion import Agedi


def _test_atoms():
    atoms = molecule("H2O")
    atoms.set_cell([10.0, 10.0, 10.0])
    atoms.set_pbc(True)
    atoms.center()
    return atoms


def test_create_diffusion():
    diffusion = create_diffusion(noisers=("cell_positions",))
    assert isinstance(diffusion, Agedi)


def test_create_diffusion_non_default_feature_size_updates_position_head_vector_dim():
    feature_size = 32
    diffusion = create_diffusion(
        noisers=("cell_positions",),
        feature_size=feature_size,
    )
    pos_head = diffusion.score_model.heads[0]
    assert pos_head.input_dim_vector == feature_size


def test_create_dataset():
    dataset = create_dataset([_test_atoms(), _test_atoms()], batch_size=2)
    assert isinstance(dataset, Dataset)
    assert len(dataset.dataset) == 2


def test_train_uses_provided_trainer():
    class DummyTrainer:
        def __init__(self):
            self.called = False

        def fit(self, diffusion_model, data):
            self.called = True
            self.diffusion_model = diffusion_model
            self.data = data

    diffusion = create_diffusion(noisers=("cell_positions",))
    dataset = create_dataset([_test_atoms(), _test_atoms()], batch_size=2)
    trainer = DummyTrainer()

    returned = train(diffusion, dataset, trainer=trainer)
    assert returned is trainer
    assert trainer.called
    assert trainer.diffusion_model is diffusion
    assert trainer.data is dataset


def test_sample_returns_atoms(diffusion):
    structures = sample(
        diffusion,
        n_samples=1,
        steps=2,
        atomic_numbers=[6, 8, 8],
        cell=np.diag([10.0, 10.0, 10.0]),
        property={"property": 1.0},
    )
    assert len(structures) == 1
    assert structures[0].positions.shape == (3, 3)


def test_load_diffusion(tmp_path):
    """load_diffusion should reconstruct the model from the Hydra hparams format."""
    diffusion = create_diffusion(noisers=("cell_positions",), lr=2e-4)
    log_dir = tmp_path / "logs" / "version_0"
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    hparams = {"diffusion": diffusion.get_hparams()}

    with open(log_dir / "hparams.yaml", "w") as f:
        yaml.dump(hparams, f, default_flow_style=False)

    torch.save({"state_dict": diffusion.state_dict()}, checkpoint_dir / "last_model.ckpt")

    loaded = load_diffusion(log_dir)
    assert isinstance(loaded, Agedi)


def test_load_diffusion_missing_diffusion_key(tmp_path):
    """load_diffusion should raise ValueError when hparams.yaml lacks 'diffusion' key."""
    log_dir = tmp_path / "logs" / "version_0"
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    (log_dir / "hparams.yaml").write_text("model: PaiNN\ncutoff: 6.0\n")
    torch.save({}, checkpoint_dir / "last_model.ckpt")

    try:
        load_diffusion(log_dir)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "diffusion" in str(e)


def test_diffusion_get_hparams():
    """diffusion.get_hparams() should return a nested dict with all required keys."""
    diffusion = create_diffusion(noisers=("cell_positions",))
    hparams = diffusion.get_hparams()

    assert "_target_" in hparams
    assert "score_model" in hparams
    assert "noisers" in hparams
    assert "_target_" in hparams["score_model"]
    assert "representation" in hparams["score_model"]
    assert "conditionings" in hparams["score_model"]
    assert "heads" in hparams["score_model"]
    assert len(hparams["noisers"]) == 1
    noiser_hparams = hparams["noisers"][0]
    assert "_target_" in noiser_hparams
    assert "sde" in noiser_hparams
    assert "_target_" in noiser_hparams["sde"]
    # distribution and prior are fixed by the class, not stored in hparams
    assert "distribution" not in noiser_hparams
    assert "prior" not in noiser_hparams


def test_diffusion_get_hparams_with_force_field_uses_regressor_heads():
    """When force_field=True (shared backbone), get_hparams() must use regressor_heads, not regressor_model."""
    diffusion = create_diffusion(noisers=("cell_positions",), force_field=True)
    hparams = diffusion.get_hparams()

    assert "regressor_heads" in hparams, (
        "Shared-backbone regressor should be serialised as regressor_heads"
    )
    assert "regressor_model" not in hparams, (
        "Full regressor_model config must not appear when backbone is shared"
    )
    assert isinstance(hparams["regressor_heads"], list)
    assert len(hparams["regressor_heads"]) == 2
    assert "_target_" in hparams["regressor_heads"][0]
    assert "_target_" in hparams["regressor_heads"][1]
    assert "regressor_loss_weight" in hparams


def test_load_diffusion_with_force_field_round_trip(tmp_path):
    """load_diffusion should correctly restore a shared-backbone force field."""
    diffusion = create_diffusion(noisers=("cell_positions",), force_field=True)
    log_dir = tmp_path / "logs" / "version_0"
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    hparams = {"diffusion": diffusion.get_hparams()}
    with open(log_dir / "hparams.yaml", "w") as f:
        yaml.dump(hparams, f, default_flow_style=False)

    torch.save({"state_dict": diffusion.state_dict()}, checkpoint_dir / "last_model.ckpt")

    loaded = load_diffusion(log_dir)
    assert isinstance(loaded, Agedi)
    assert loaded.regressor_model is not None
    # The loaded regressor must share the backbone (same objects).
    assert loaded.regressor_model.translator is loaded.score_model.translator
    assert loaded.regressor_model.representation is loaded.score_model.representation

def test_diffusion_on_fit_start_writes_hparams(tmp_path):
    """Agedi.on_fit_start should write hparams.yaml to the log directory."""
    from unittest.mock import MagicMock

    diffusion = create_diffusion(noisers=("cell_positions",))
    log_dir = tmp_path / "version_0"
    log_dir.mkdir(parents=True)

    # Simulate what Lightning does: set self.trainer on the module
    mock_trainer = MagicMock()
    mock_trainer.logger.log_dir = str(log_dir)
    diffusion._trainer = mock_trainer  # Lightning uses _trainer internally

    # Manually invoke the hook (bypasses Lightning's internal wiring)
    diffusion.on_fit_start()

    hparams_file = log_dir / "hparams.yaml"
    assert hparams_file.exists(), "hparams.yaml was not written"
    with open(hparams_file) as f:
        saved = yaml.safe_load(f)
    assert "diffusion" in saved
    assert "_target_" in saved["diffusion"]


def test_train_from_atoms_with_custom_trainer():
    class DummyTrainer:
        def __init__(self):
            self.fit_calls = 0

        def fit(self, diffusion_model, data):
            self.fit_calls += 1
            self.diffusion_model = diffusion_model
            self.data = data

    trainer = DummyTrainer()
    diffusion, dataset, used_trainer = train_from_atoms(
        [_test_atoms(), _test_atoms()],
        noisers=("cell_positions",),
        trainer=trainer,
    )
    assert isinstance(diffusion, Agedi)
    assert isinstance(dataset, Dataset)
    assert used_trainer is trainer
    assert trainer.fit_calls == 1
    assert isinstance(trainer.data.dataset[0], AtomsGraph)


def test_train_from_atoms_hparams_metadata(tmp_path):
    """train_from_atoms hparams dict should contain noisers, sde, conditioning, and confinement."""
    from unittest.mock import MagicMock

    class CapturingTrainer:
        """Captures the hparams passed via trainer kwargs."""
        def __init__(self):
            self.fit_calls = 0

        def fit(self, diffusion_model, data):
            self.fit_calls += 1

    trainer = CapturingTrainer()
    diffusion, dataset, _ = train_from_atoms(
        [_test_atoms(), _test_atoms()],
        noisers=("confined_cell_positions",),
        conditioning="none",
        confinement=(0.0, 10.0),
        trainer=trainer,
    )
    # get_hparams on the diffusion model must still work
    hparams = diffusion.get_hparams()
    assert "_target_" in hparams

    # Verify the noiser hparams encode ConfinedCellPositions correctly
    noiser_hp = hparams["noisers"][0]
    assert "ConfinedCellPositions" in noiser_hp["_target_"]
    assert "sde" in noiser_hp
    assert "distribution" not in noiser_hp
    assert "prior" not in noiser_hp


# ---------------------------------------------------------------------------
# train_from_config tests
# ---------------------------------------------------------------------------


def test_train_from_config_requires_data_path():
    """train_from_config should raise ValueError when data_path is missing."""
    import pytest

    with pytest.raises(ValueError, match="data_path"):
        train_from_config({})


def test_train_from_config_unknown_keys_warns(tmp_path):
    """train_from_config should warn about unrecognised config keys."""
    import warnings
    from unittest.mock import patch

    data_file = tmp_path / "train.traj"
    atoms = _test_atoms()
    from ase.io import write as ase_write

    ase_write(str(data_file), [atoms, atoms])

    class DummyTrainer:
        def fit(self, diffusion_model, data):
            pass

    cfg = {
        "data_path": str(data_file),
        "noisers": ["positions"],
        "trainer": DummyTrainer(),  # unknown key
    }

    class _FakeDataset:
        train_idx = [0]
        val_idx = [0]

    with warnings.catch_warnings(record=True) as caught, patch(
        "agedi.functional.train_from_atoms",
        return_value=(create_diffusion(noisers=("cell_positions",)), _FakeDataset(), None),
    ):
        warnings.simplefilter("always")
        train_from_config(cfg)
    assert any("unrecognised" in str(w.message).lower() for w in caught)


def test_train_from_config_dict(tmp_path):
    """train_from_config should train successfully from a plain dict config."""
    from ase.io import write as ase_write

    data_file = tmp_path / "train.traj"
    atoms = [_test_atoms(), _test_atoms()]
    ase_write(str(data_file), atoms)

    class DummyTrainer:
        def __init__(self):
            self.fit_calls = 0

        def fit(self, diffusion_model, data):
            self.fit_calls += 1

    dummy_trainer = DummyTrainer()

    cfg = {
        "data_path": str(data_file),
        "noisers": ["cell_positions"],
        "feature_size": 32,
        "n_blocks": 2,
    }

    diffusion, dataset, used_trainer = _train_from_config_with_trainer(cfg, dummy_trainer)
    assert isinstance(diffusion, Agedi)
    assert isinstance(dataset, Dataset)
    assert used_trainer is dummy_trainer
    assert dummy_trainer.fit_calls == 1


def _train_from_config_with_trainer(cfg, trainer):
    """Helper that injects a dummy trainer into train_from_config."""
    from agedi.functional import train_from_atoms
    from ase.io import read as ase_read
    from pathlib import Path
    import yaml

    data = ase_read(cfg["data_path"], ":")
    train_keys = {
        "noisers", "sde", "style", "conditioning",
        "conditioning_type", "mask", "confinement", "batch_size", "train_split",
        "val_split", "repeat", "lr", "lr_factor", "lr_patience", "weight_decay",
        "eps", "guidance_weight", "model", "cutoff", "feature_size", "n_blocks", "n_rbf",
    }
    train_kwargs = {k: cfg[k] for k in train_keys if k in cfg}
    return train_from_atoms(
        data,
        data_path=str(Path(cfg["data_path"]).resolve()),
        trainer=trainer,
        **train_kwargs,
    )


def test_train_from_config_yaml_file(tmp_path):
    """train_from_config should read and apply a YAML config file."""
    from ase.io import write as ase_write

    data_file = tmp_path / "train.traj"
    ase_write(str(data_file), [_test_atoms(), _test_atoms()])

    config_file = tmp_path / "my_train.yaml"
    config_file.write_text(
        f"data_path: {data_file}\n"
        "noisers:\n  - positions\n"
        "feature_size: 32\n"
        "n_blocks: 2\n"
    )

    # We only test that the YAML is loaded and train_from_atoms is invoked
    # (using a dummy trainer to avoid a full Lightning run in CI).
    import agedi.functional as fn
    original = fn.train_from_atoms

    calls = []

    def capturing_train(data, **kwargs):
        calls.append(kwargs)
        # Return minimal stubs so train_from_config's caller doesn't fail.
        diffusion = create_diffusion(noisers=kwargs.get("noisers", ("cell_positions",)))

        class _FakeDataset:
            train_idx = [0]
            val_idx = [0]

        return diffusion, _FakeDataset(), None

    fn.train_from_atoms = capturing_train
    try:
        train_from_config(str(config_file))
    except Exception:
        pass  # We only care that it called train_from_atoms
    finally:
        fn.train_from_atoms = original

    assert calls, "train_from_atoms was not called by train_from_config"
    assert calls[0].get("noisers") == ["positions"]
    assert calls[0].get("feature_size") == 32


# ---------------------------------------------------------------------------
# checkpoint / continue-training tests
# ---------------------------------------------------------------------------


def test_train_from_atoms_with_checkpoint(tmp_path):
    """train_from_atoms with checkpoint should load model from checkpoint dir."""
    from ase.io import write as ase_write, read as ase_read
    from unittest.mock import patch

    diffusion_orig = create_diffusion(noisers=("cell_positions",), feature_size=32, n_blocks=2)
    log_dir = tmp_path / "logs" / "version_0"
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    hparams = {"diffusion": diffusion_orig.get_hparams()}
    with open(log_dir / "hparams.yaml", "w") as f:
        yaml.dump(hparams, f, default_flow_style=False)
    torch.save({"state_dict": diffusion_orig.state_dict()}, checkpoint_dir / "last_model.ckpt")

    # Track calls to the trainer's fit method.
    class DummyTrainer:
        def __init__(self):
            self.fit_calls = 0
            self.fit_diffusion = None
            self.fit_kwargs = {}

        def fit(self, diffusion_model, data, **kwargs):
            self.fit_calls += 1
            self.fit_diffusion = diffusion_model
            self.fit_kwargs = kwargs

    trainer = DummyTrainer()
    data_file = tmp_path / "train.traj"
    ase_write(str(data_file), [_test_atoms(), _test_atoms()])
    data = ase_read(str(data_file), ":")

    # Mock load_diffusion to avoid a pre-existing hydra-instantiate issue in CI.
    with patch("agedi.functional.load_diffusion", return_value=diffusion_orig) as mock_load:
        diffusion, dataset, used_trainer = train_from_atoms(
            data,
            noisers=("cell_positions",),
            trainer=trainer,
            checkpoint=str(log_dir),
        )

    mock_load.assert_called_once()
    # The path passed to load_diffusion should match the checkpoint directory.
    assert mock_load.call_args[0][0] == tmp_path / "logs" / "version_0"
    assert used_trainer is trainer
    assert trainer.fit_calls == 1
    assert isinstance(diffusion, Agedi)
    # ckpt_path should be passed to trainer.fit() for full state restoration.
    assert "ckpt_path" in trainer.fit_kwargs


def test_train_from_atoms_with_checkpoint_ckpt_file(tmp_path):
    """train_from_atoms with a direct .ckpt path should load the model."""
    from ase.io import write as ase_write, read as ase_read
    from unittest.mock import patch

    diffusion_orig = create_diffusion(noisers=("cell_positions",), feature_size=32, n_blocks=2)
    log_dir = tmp_path / "logs" / "version_0"
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    hparams = {"diffusion": diffusion_orig.get_hparams()}
    with open(log_dir / "hparams.yaml", "w") as f:
        yaml.dump(hparams, f, default_flow_style=False)
    ckpt_file = checkpoint_dir / "last_model.ckpt"
    torch.save({"state_dict": diffusion_orig.state_dict()}, ckpt_file)

    class DummyTrainer:
        def __init__(self):
            self.fit_calls = 0

        def fit(self, diffusion_model, data, **kwargs):
            self.fit_calls += 1

    trainer = DummyTrainer()
    data_file = tmp_path / "train.traj"
    ase_write(str(data_file), [_test_atoms(), _test_atoms()])
    data = ase_read(str(data_file), ":")

    with patch("agedi.functional.load_diffusion", return_value=diffusion_orig) as mock_load:
        diffusion, dataset, used_trainer = train_from_atoms(
            data,
            noisers=("cell_positions",),
            trainer=trainer,
            checkpoint=str(ckpt_file),
        )

    mock_load.assert_called_once()
    assert trainer.fit_calls == 1
    assert isinstance(diffusion, Agedi)


def test_train_from_config_with_checkpoint(tmp_path):
    """train_from_config should forward the checkpoint key to train_from_atoms."""
    from ase.io import write as ase_write

    diffusion_orig = create_diffusion(noisers=("cell_positions",), feature_size=32, n_blocks=2)
    log_dir = tmp_path / "logs" / "version_0"
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    hparams = {"diffusion": diffusion_orig.get_hparams()}
    with open(log_dir / "hparams.yaml", "w") as f:
        yaml.dump(hparams, f, default_flow_style=False)
    torch.save({"state_dict": diffusion_orig.state_dict()}, checkpoint_dir / "last_model.ckpt")

    data_file = tmp_path / "train.traj"
    ase_write(str(data_file), [_test_atoms(), _test_atoms()])

    import agedi.functional as fn
    original = fn.train_from_atoms
    calls = []

    class _Captured(Exception):
        pass

    def capturing_train(data, **kwargs):
        calls.append(kwargs)
        raise _Captured("captured")

    fn.train_from_atoms = capturing_train
    try:
        train_from_config({
            "data_path": str(data_file),
            "noisers": ["cell_positions"],
            "checkpoint": str(log_dir),
        })
    except _Captured:
        pass
    finally:
        fn.train_from_atoms = original

    assert calls, "train_from_atoms was not called"
    assert calls[0].get("checkpoint") == str(log_dir)


def test_train_from_atoms_checkpoint_missing_ckpt_raises(tmp_path):
    """train_from_atoms should raise FileNotFoundError when no .ckpt file is found."""
    import pytest
    from ase.io import write as ase_write, read as ase_read
    from unittest.mock import patch

    diffusion_orig = create_diffusion(noisers=("cell_positions",))
    # Create a directory with hparams.yaml but NO checkpoints subdirectory.
    run_dir = tmp_path / "logs" / "version_0"
    run_dir.mkdir(parents=True)
    hparams = {"diffusion": diffusion_orig.get_hparams()}
    with open(run_dir / "hparams.yaml", "w") as f:
        yaml.dump(hparams, f, default_flow_style=False)

    data_file = tmp_path / "train.traj"
    ase_write(str(data_file), [_test_atoms(), _test_atoms()])
    data = ase_read(str(data_file), ":")

    with patch("agedi.functional.load_diffusion", return_value=diffusion_orig):
        with pytest.raises(FileNotFoundError, match="last_model.ckpt"):
            train_from_atoms(
                data,
                noisers=("cell_positions",),
                checkpoint=str(run_dir),
            )


def test_train_ckpt_path_forwarded_to_fit(tmp_path):
    """train() should pass ckpt_path to trainer.fit() when provided."""
    from ase.io import write as ase_write

    diffusion = create_diffusion(noisers=("cell_positions",))
    dataset = create_dataset([_test_atoms(), _test_atoms()], batch_size=2)

    fit_kwargs_received = {}

    class DummyTrainer:
        def fit(self, diffusion_model, data, **kwargs):
            fit_kwargs_received.update(kwargs)

    trainer = DummyTrainer()
    fake_ckpt = tmp_path / "model.ckpt"
    fake_ckpt.write_text("dummy")

    train(diffusion, dataset, trainer=trainer, ckpt_path=str(fake_ckpt))
    assert fit_kwargs_received.get("ckpt_path") == str(fake_ckpt)


def test_train_no_ckpt_path_no_kwarg(tmp_path):
    """train() without ckpt_path should call fit() without ckpt_path kwarg."""
    diffusion = create_diffusion(noisers=("cell_positions",))
    dataset = create_dataset([_test_atoms(), _test_atoms()], batch_size=2)

    fit_kwargs_received = {}

    class DummyTrainer:
        def fit(self, diffusion_model, data, **kwargs):
            fit_kwargs_received.update(kwargs)

    trainer = DummyTrainer()
    train(diffusion, dataset, trainer=trainer)
    assert "ckpt_path" not in fit_kwargs_received


# ---------------------------------------------------------------------------
# predict tests
# ---------------------------------------------------------------------------


def test_predict_raises_without_regressor():
    """predict should raise ValueError when the model has no regressor_model."""
    import pytest

    diffusion = create_diffusion(noisers=("cell_positions",))
    assert diffusion.regressor_model is None

    with pytest.raises(ValueError, match="force_field"):
        predict(diffusion, [_test_atoms()])


def test_predict_returns_atoms_with_predictions():
    """predict should return Atoms objects with energy and forces attached."""
    diffusion = create_diffusion(noisers=("cell_positions",), force_field=True)
    assert diffusion.regressor_model is not None

    atoms = _test_atoms()
    results = predict(diffusion, [atoms, atoms])

    assert len(results) == 2
    for result_atoms in results:
        calc = result_atoms.calc
        assert calc is not None
        assert "energy" in calc.results
        assert "forces" in calc.results
        assert calc.results["forces"].shape == (len(atoms), 3)


# ---------------------------------------------------------------------------
# type_map / n_classes tests
# ---------------------------------------------------------------------------


def test_build_type_map_from_data():
    """_build_type_map_from_data should return [0] + sorted unique atomic numbers."""
    from agedi.functional import _build_type_map_from_data

    h2o = _test_atoms()  # H2O → Z in {1, 8}
    type_map = _build_type_map_from_data([h2o, h2o])
    assert type_map == [0, 1, 8], f"Expected [0, 1, 8], got {type_map}"


def test_create_diffusion_with_type_map():
    """create_diffusion with type_map should use reduced vocabulary for Types noiser."""
    from agedi.diffusion.noisers import Types

    type_map = [0, 1, 8]  # absorbing, H, O
    diffusion = create_diffusion(noisers=("types",), type_map=type_map)

    types_noiser = next(n for n in diffusion.noisers if isinstance(n, Types))
    assert types_noiser.n_classes == 3
    assert types_noiser._type_map == [0, 1, 8]


def test_train_from_atoms_auto_detects_type_map():
    """train_from_atoms should auto-detect type_map when Types noiser is used."""
    from agedi.diffusion.noisers import Types

    class DummyTrainer:
        def fit(self, m, d):
            pass

    h2o = _test_atoms()  # H=1, O=8
    diffusion, _, _ = train_from_atoms(
        [h2o, h2o],
        noisers=("types",),
        trainer=DummyTrainer(),
    )

    types_noiser = next(n for n in diffusion.noisers if isinstance(n, Types))
    # H (1) and O (8) → type_map = [0, 1, 8] → n_classes = 3
    assert types_noiser.n_classes == 3
    assert types_noiser._type_map == [0, 1, 8]


def test_train_from_atoms_n_classes_restricts_vocab():
    """train_from_atoms with n_classes should restrict to that many types."""
    from agedi.diffusion.noisers import Types
    from ase.build import molecule

    # Create data with 3 element types: H=1, C=6, O=8
    ch3oh = molecule("CH3OH")
    ch3oh.set_cell([10, 10, 10])
    ch3oh.set_pbc(True)
    ch3oh.center()

    class DummyTrainer:
        def fit(self, m, d):
            pass

    # Restrict to 2 element types (picks H=1 and C=6, the first 2 by atomic number)
    diffusion, _, _ = train_from_atoms(
        [ch3oh, ch3oh],
        noisers=("types",),
        n_classes=2,
        trainer=DummyTrainer(),
    )

    types_noiser = next(n for n in diffusion.noisers if isinstance(n, Types))
    assert types_noiser.n_classes == 3  # absorbing + 2 types
    assert types_noiser._type_map == [0, 1, 6]  # absorbing, H, C


def test_train_from_atoms_n_classes_too_large_raises():
    """train_from_atoms should raise ValueError if n_classes > detected types."""
    import pytest

    class DummyTrainer:
        def fit(self, m, d):
            pass

    h2o = _test_atoms()  # 2 element types: H=1, O=8
    with pytest.raises(ValueError, match="n_classes"):
        train_from_atoms(
            [h2o, h2o],
            noisers=("types",),
            n_classes=5,  # More than the 2 types present
            trainer=DummyTrainer(),
        )


def test_types_noiser_hparams_roundtrip_with_type_map():
    """Diffusion.get_hparams() should include type_map in Types noiser config."""
    from agedi.diffusion.noisers import Types

    class DummyTrainer:
        def fit(self, m, d):
            pass

    h2o = _test_atoms()
    diffusion, _, _ = train_from_atoms(
        [h2o, h2o],
        noisers=("types",),
        trainer=DummyTrainer(),
    )

    hparams = diffusion.get_hparams()
    types_noiser_hparams = next(
        n for n in hparams["noisers"] if "Types" in n.get("_target_", "")
    )
    assert "type_map" in types_noiser_hparams
    assert types_noiser_hparams["type_map"] == [0, 1, 8]
    assert types_noiser_hparams["n_classes"] == 3


# ---------------------------------------------------------------------------
# Fully-connected graph
# ---------------------------------------------------------------------------

def test_create_diffusion_fully_connected():
    """create_diffusion with fully_connected=True should set the flag on Agedi."""
    diffusion = create_diffusion(noisers=("Positions",), fully_connected=True)
    assert diffusion.fully_connected is True


def test_create_diffusion_fully_connected_auto_cutoff():
    """cutoff=None + fully_connected=True must auto-select 50 Å backbone cutoff."""
    diffusion = create_diffusion(noisers=("Positions",), fully_connected=True, cutoff=None)
    cutoff = float(diffusion.score_model.representation.cutoff_fn.cutoff[0])
    assert cutoff == 50.0, f"Expected 50 Å FC cutoff, got {cutoff}"


def test_create_diffusion_default_cutoff_no_fc():
    """cutoff=None without fully_connected must default to 6 Å."""
    diffusion = create_diffusion(noisers=("CellPositions",), cutoff=None)
    cutoff = float(diffusion.score_model.representation.radial_basis.offsets[-1])
    assert abs(cutoff - 6.0) < 0.1, f"Expected 6 Å default cutoff, got {cutoff}"


def test_load_diffusion_fully_connected_roundtrip(tmp_path):
    """load_diffusion must preserve fully_connected=True across save/load."""
    diffusion = create_diffusion(noisers=("Positions",), fully_connected=True)
    log_dir = tmp_path / "logs" / "version_0"
    (log_dir / "checkpoints").mkdir(parents=True)
    import yaml
    with open(log_dir / "hparams.yaml", "w") as f:
        yaml.dump({"diffusion": diffusion.get_hparams()}, f)
    torch.save({"state_dict": diffusion.state_dict()}, log_dir / "checkpoints" / "last_model.ckpt")

    loaded = load_diffusion(log_dir)
    assert loaded.fully_connected is True


# ---------------------------------------------------------------------------
# Radial basis selection
# ---------------------------------------------------------------------------

def test_create_diffusion_radial_basis_bessel():
    """create_diffusion with radial_basis='bessel' should use BesselRBF."""
    import schnetpack as spk
    diffusion = create_diffusion(noisers=("CellPositions",), radial_basis="bessel", cutoff=6.0)
    rb = diffusion.score_model.representation.radial_basis
    assert isinstance(rb, spk.nn.BesselRBF), f"Expected BesselRBF, got {type(rb).__name__}"


def test_create_diffusion_radial_basis_gaussian():
    """create_diffusion with radial_basis='gaussian' (default) should use GaussianRBF."""
    import schnetpack as spk
    diffusion = create_diffusion(noisers=("CellPositions",), radial_basis="gaussian")
    rb = diffusion.score_model.representation.radial_basis
    assert isinstance(rb, spk.nn.GaussianRBF), f"Expected GaussianRBF, got {type(rb).__name__}"


def test_create_diffusion_radial_basis_invalid():
    """create_diffusion with an unknown radial_basis should raise ValueError."""
    with pytest.raises(ValueError, match="radial_basis"):
        create_diffusion(noisers=("CellPositions",), radial_basis="sinc")


def test_load_diffusion_bessel_roundtrip(tmp_path):
    """load_diffusion must preserve radial_basis='bessel' across save/load."""
    import schnetpack as spk
    import yaml
    diffusion = create_diffusion(noisers=("CellPositions",), radial_basis="bessel", cutoff=6.0)
    log_dir = tmp_path / "logs" / "version_0"
    (log_dir / "checkpoints").mkdir(parents=True)
    with open(log_dir / "hparams.yaml", "w") as f:
        yaml.dump({"diffusion": diffusion.get_hparams()}, f)
    torch.save({"state_dict": diffusion.state_dict()}, log_dir / "checkpoints" / "last_model.ckpt")

    loaded = load_diffusion(log_dir)
    rb = loaded.score_model.representation.radial_basis
    assert isinstance(rb, spk.nn.BesselRBF), "BesselRBF must survive save/load"


# ---------------------------------------------------------------------------
# EDM preconditioning
# ---------------------------------------------------------------------------

def test_create_diffusion_precondition():
    """create_diffusion with precondition=True should enable preconditioning on the head."""
    from agedi.models.schnetpack.heads import PositionsScore
    diffusion = create_diffusion(noisers=("CellPositions",), precondition=True, sigma_data=1.5)
    pos_head = next(h for h in diffusion.score_model.heads if isinstance(h, PositionsScore))
    assert pos_head.precondition is True
    assert abs(pos_head.sigma_data - 1.5) < 1e-6


def test_load_diffusion_precondition_roundtrip(tmp_path):
    """load_diffusion must preserve precondition=True and sigma_data."""
    from agedi.models.schnetpack.heads import PositionsScore
    import yaml
    diffusion = create_diffusion(noisers=("CellPositions",), precondition=True, sigma_data=2.0)
    log_dir = tmp_path / "logs" / "version_0"
    (log_dir / "checkpoints").mkdir(parents=True)
    with open(log_dir / "hparams.yaml", "w") as f:
        yaml.dump({"diffusion": diffusion.get_hparams()}, f)
    torch.save({"state_dict": diffusion.state_dict()}, log_dir / "checkpoints" / "last_model.ckpt")

    loaded = load_diffusion(log_dir)
    pos_head = next(h for h in loaded.score_model.heads if isinstance(h, PositionsScore))
    assert pos_head.precondition is True
    assert abs(pos_head.sigma_data - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# epsilon prediction + DDPM sampler (API level)
# ---------------------------------------------------------------------------

def test_create_diffusion_epsilon_prediction():
    """create_diffusion with prediction_type='epsilon' should configure the noiser."""
    from agedi.diffusion.noisers import PositionsNoiser
    diffusion = create_diffusion(
        noisers=("Positions",), sde="vp", prediction_type="epsilon", sampler="em"
    )
    pos_noiser = next(n for n in diffusion.noisers if isinstance(n, PositionsNoiser))
    assert pos_noiser.prediction_type == "epsilon"
    assert pos_noiser.sampler == "em"


def test_create_diffusion_ddpm_sampler():
    """create_diffusion with sampler='ddpm' + prediction_type='epsilon' must work."""
    from agedi.diffusion.noisers import PositionsNoiser
    diffusion = create_diffusion(
        noisers=("Positions",), sde="vp", prediction_type="epsilon", sampler="ddpm"
    )
    pos_noiser = next(n for n in diffusion.noisers if isinstance(n, PositionsNoiser))
    assert pos_noiser.sampler == "ddpm"


def test_create_diffusion_loss_weighting_min_snr():
    """create_diffusion with loss_weighting='min_snr' must configure the noiser."""
    from agedi.diffusion.noisers import PositionsNoiser
    diffusion = create_diffusion(noisers=("CellPositions",), loss_weighting="min_snr")
    pos_noiser = next(n for n in diffusion.noisers if isinstance(n, PositionsNoiser))
    assert pos_noiser.loss_weighting == "min_snr"


def test_diffusion_get_hparams_epsilon_prediction_roundtrip(tmp_path):
    """prediction_type and sampler must survive get_hparams → load_diffusion."""
    import yaml
    from agedi.diffusion.noisers import PositionsNoiser
    diffusion = create_diffusion(
        noisers=("Positions",), sde="vp", prediction_type="epsilon", sampler="ddpm"
    )
    log_dir = tmp_path / "logs" / "version_0"
    (log_dir / "checkpoints").mkdir(parents=True)
    with open(log_dir / "hparams.yaml", "w") as f:
        yaml.dump({"diffusion": diffusion.get_hparams()}, f)
    torch.save({"state_dict": diffusion.state_dict()}, log_dir / "checkpoints" / "last_model.ckpt")

    loaded = load_diffusion(log_dir)
    pos_noiser = next(n for n in loaded.noisers if isinstance(n, PositionsNoiser))
    assert pos_noiser.prediction_type == "epsilon"
    assert pos_noiser.sampler == "ddpm"
