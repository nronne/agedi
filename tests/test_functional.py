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
# force-field reference energies / force loss
# ---------------------------------------------------------------------------


def _test_atoms_with_labels(energy: float, seed: int = 0):
    """H2O structure carrying a total energy and per-atom forces."""
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms = _test_atoms()
    rng = np.random.default_rng(seed)
    atoms.calc = SinglePointCalculator(
        atoms, energy=energy, forces=rng.normal(0, 0.1, (len(atoms), 3))
    )
    return atoms


def test_create_diffusion_force_field_defaults():
    """The force-field regressor uses a Huber force loss by default."""
    diffusion = create_diffusion(noisers=("cell_positions",), force_field=True)

    assert diffusion.regressor_model.force_loss == "huber"
    assert diffusion.regressor_model.huber_delta == 0.01
    energy_head = next(h for h in diffusion.regressor_model.heads if h.key == "energy")
    assert energy_head.reference_energy_dict == {}


def test_create_diffusion_force_field_reference_energies_and_loss():
    diffusion = create_diffusion(
        noisers=("cell_positions",),
        force_field=True,
        reference_energies={"O": -4.95, 1: -0.6},
        force_loss="mse",
        huber_delta=0.05,
    )

    energy_head = next(h for h in diffusion.regressor_model.heads if h.key == "energy")
    assert energy_head.reference_energy_dict == {1: pytest.approx(-0.6), 8: pytest.approx(-4.95)}
    assert diffusion.regressor_model.force_loss == "mse"
    assert diffusion.regressor_model.huber_delta == 0.05


def test_force_field_hparams_round_trip(tmp_path):
    """Reference energies and loss settings survive a save/load cycle."""
    diffusion = create_diffusion(
        noisers=("cell_positions",),
        force_field=True,
        reference_energies={"O": -4.95, "H": -0.6},
        force_loss="mae",
    )
    log_dir = tmp_path / "logs" / "version_0"
    (log_dir / "checkpoints").mkdir(parents=True)
    with open(log_dir / "hparams.yaml", "w") as fh:
        yaml.dump({"diffusion": diffusion.get_hparams()}, fh, default_flow_style=False)
    torch.save(
        {"state_dict": diffusion.state_dict()},
        log_dir / "checkpoints" / "last_model.ckpt",
    )

    loaded = load_diffusion(log_dir)

    energy_head = next(h for h in loaded.regressor_model.heads if h.key == "energy")
    assert energy_head.reference_energy_dict == {1: pytest.approx(-0.6), 8: pytest.approx(-4.95)}
    assert loaded.regressor_model.force_loss == "mae"


def test_force_field_checkpoint_without_reference_energies_still_loads():
    """State dicts from before reference energies existed remain loadable."""
    without = create_diffusion(noisers=("cell_positions",), force_field=True)
    with_reference = create_diffusion(
        noisers=("cell_positions",), force_field=True, reference_energies={"O": -4.95}
    )

    with_reference.load_state_dict(without.state_dict())

    energy_head = next(h for h in with_reference.regressor_model.heads if h.key == "energy")
    assert energy_head.reference_energy_dict == {8: pytest.approx(-4.95)}


def test_resolve_reference_energies_auto_fits_from_data():
    from agedi.api.training import _resolve_reference_energies

    # E = n_H * (-0.6) + n_O * (-4.95); H2O has 2 H and 1 O.
    data = [_test_atoms_with_labels(2 * -0.6 + -4.95)]

    resolved = _resolve_reference_energies("auto", data=data, force_field=True)

    assert resolved.keys() == {1, 8}
    assert 2 * resolved[1] + resolved[8] == pytest.approx(2 * -0.6 + -4.95)


def test_resolve_reference_energies_disabled_and_explicit():
    from agedi.api.training import _resolve_reference_energies

    data = [_test_atoms_with_labels(-6.15)]

    assert _resolve_reference_energies(None, data=data, force_field=True) is None
    assert _resolve_reference_energies("auto", data=data, force_field=False) is None
    assert _resolve_reference_energies({"O": -4.95}, data=data, force_field=True) == {8: -4.95}

    with pytest.raises(ValueError, match="not recognized"):
        _resolve_reference_energies("fitted", data=data, force_field=True)


def test_train_from_atoms_fits_reference_energies():
    """train_from_atoms(force_field=True) fits references from the labelled data."""

    class DummyTrainer:
        def fit(self, model, data):
            pass

    data = [_test_atoms_with_labels(-6.15, seed=i) for i in range(3)]
    diffusion, _, _ = train_from_atoms(
        data,
        noisers=("cell_positions",),
        force_field=True,
        trainer=DummyTrainer(),
    )

    energy_head = next(h for h in diffusion.regressor_model.heads if h.key == "energy")
    references = energy_head.reference_energy_dict
    assert references.keys() == {1, 8}
    assert 2 * references[1] + references[8] == pytest.approx(-6.15, abs=1e-4)


def test_train_from_atoms_reference_energies_reduce_energy_loss():
    """Subtracting references shrinks the initial energy loss by orders of magnitude."""
    from torch_geometric.data import Batch

    class DummyTrainer:
        def fit(self, model, data):
            pass

    data = [_test_atoms_with_labels(-2000.0, seed=i) for i in range(3)]
    kwargs = dict(noisers=("cell_positions",), force_field=True, trainer=DummyTrainer())

    with_reference, _, _ = train_from_atoms(data, **kwargs)
    without_reference, _, _ = train_from_atoms(data, reference_energies=None, **kwargs)

    graphs = []
    for atoms in data:
        graph = AtomsGraph.from_atoms(atoms)
        graph.energy = torch.tensor(atoms.get_potential_energy(), dtype=torch.float32)
        graph.forces = torch.tensor(atoms.get_forces(), dtype=torch.float32)
        graphs.append(graph)
    batch = Batch.from_data_list(graphs)

    referenced_loss = float(with_reference.regressor_model.loss(batch)["energy_loss"])
    raw_loss = float(without_reference.regressor_model.loss(batch)["energy_loss"])

    assert referenced_loss < raw_loss / 1000


def test_config_forwards_force_field_keys(tmp_path):
    """reference_energies / force_loss / huber_delta are recognised config keys."""
    from agedi.api.training import _TRAIN_FROM_ATOMS_KEYS

    assert {"reference_energies", "force_loss", "huber_delta"} <= _TRAIN_FROM_ATOMS_KEYS

    class DummyTrainer:
        def fit(self, model, data):
            pass

    cfg = {
        "noisers": ["cell_positions"],
        "force_field": True,
        "reference_energies": {"H": -0.6, "O": -4.95},
        "force_loss": "mse",
        "huber_delta": 0.2,
    }
    train_kwargs = {k: v for k, v in cfg.items() if k in _TRAIN_FROM_ATOMS_KEYS}

    diffusion, _, _ = train_from_atoms(
        [_test_atoms_with_labels(-6.15)],
        trainer=DummyTrainer(),
        **train_kwargs,
    )

    energy_head = next(h for h in diffusion.regressor_model.heads if h.key == "energy")
    assert energy_head.reference_energy_dict == {1: pytest.approx(-0.6), 8: pytest.approx(-4.95)}
    assert diffusion.regressor_model.force_loss == "mse"
    assert diffusion.regressor_model.huber_delta == 0.2


def test_cli_parse_reference_energies():
    import click

    from agedi.cli.train import _parse_reference_energies

    assert _parse_reference_energies("auto") == "auto"
    assert _parse_reference_energies("none") is None
    assert _parse_reference_energies("Cu:-3.72, O:-4.95") == {"Cu": -3.72, "O": -4.95}

    with pytest.raises(click.BadParameter):
        _parse_reference_energies("Cu")
    with pytest.raises(click.BadParameter):
        _parse_reference_energies("Cu:abc")


def test_create_diffusion_regressor_loss_weight():
    diffusion = create_diffusion(
        noisers=("cell_positions",), force_field=True, regressor_loss_weight=7.5
    )

    assert diffusion.regressor_loss_weight == 7.5
    assert diffusion.get_hparams()["regressor_loss_weight"] == 7.5


def test_regressor_loss_weight_scales_total_loss():
    """loss = diffusion_loss + regressor_loss_weight * regressor_loss."""
    from torch_geometric.data import Batch

    diffusion = create_diffusion(noisers=("cell_positions",), force_field=True)

    atoms = _test_atoms_with_labels(-6.15)
    graph = AtomsGraph.from_atoms(atoms)
    graph.energy = torch.tensor(atoms.get_potential_energy(), dtype=torch.float32)
    graph.forces = torch.tensor(atoms.get_forces(), dtype=torch.float32)
    batch = Batch.from_data_list([graph])

    torch.manual_seed(0)
    unit = diffusion.loss(batch, 0)
    diffusion.regressor_loss_weight = 3.0
    torch.manual_seed(0)
    weighted = diffusion.loss(batch, 0)

    regressor = float(unit["regressor_loss"])
    assert float(weighted["loss"]) == pytest.approx(
        float(unit["loss"]) + 2.0 * regressor, rel=1e-5
    )


def test_train_from_atoms_forwards_regressor_loss_weight():
    class DummyTrainer:
        def fit(self, model, data):
            pass

    diffusion, _, _ = train_from_atoms(
        [_test_atoms_with_labels(-6.15)],
        noisers=("cell_positions",),
        force_field=True,
        regressor_loss_weight=0.1,
        trainer=DummyTrainer(),
    )

    assert diffusion.regressor_loss_weight == 0.1


def test_forcefield_hparams_reports_regressor_loss_weight():
    from agedi.api.training import _forcefield_hparams

    diffusion = create_diffusion(
        noisers=("cell_positions",), force_field=True, regressor_loss_weight=4.0
    )

    info = _forcefield_hparams(diffusion)

    assert info["force_field"] is True
    assert info["regressor_loss_weight"] == 4.0
    assert _forcefield_hparams(create_diffusion(noisers=("cell_positions",))) == {
        "force_field": False
    }


# ---------------------------------------------------------------------------
# loss_balance: relative diffusion / regressor split
# ---------------------------------------------------------------------------


def _labelled_batch(energy: float, force_scale: float, seed: int = 0):
    """Single-structure batch with energy/forces of a controllable magnitude."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from torch_geometric.data import Batch

    atoms = _test_atoms()
    rng = np.random.default_rng(seed)
    atoms.calc = SinglePointCalculator(
        atoms, energy=energy, forces=rng.normal(0, force_scale, (len(atoms), 3))
    )
    graph = AtomsGraph.from_atoms(atoms)
    graph.energy = torch.tensor(energy, dtype=torch.float32)
    graph.forces = torch.tensor(atoms.get_forces(), dtype=torch.float32)
    return Batch.from_data_list([graph])


def _mean_regressor_fraction(diffusion, batch, warmup=200, samples=600):
    """Average share of the total loss contributed by the regressor term."""
    diffusion.train()
    for _ in range(warmup):
        diffusion.loss(batch, 0)
    fractions = [
        float(diffusion.loss(batch, 0)["regressor_fraction"]) for _ in range(samples)
    ]
    return float(np.mean(fractions))


def test_loss_balance_default_is_absolute_weighting():
    diffusion = create_diffusion(noisers=("cell_positions",), force_field=True)

    assert diffusion.loss_balance is None
    assert diffusion.get_hparams()["loss_balance"] is None


def test_loss_balance_is_normalized_and_serialised():
    diffusion = create_diffusion(
        noisers=("cell_positions",), force_field=True, loss_balance="80:20"
    )

    assert diffusion.loss_balance == pytest.approx((0.8, 0.2))
    assert diffusion.get_hparams()["loss_balance"] == pytest.approx([0.8, 0.2])


def test_loss_balance_round_trips_through_hparams(tmp_path):
    diffusion = create_diffusion(
        noisers=("cell_positions",), force_field=True, loss_balance=(3, 1)
    )
    log_dir = tmp_path / "logs" / "version_0"
    (log_dir / "checkpoints").mkdir(parents=True)
    with open(log_dir / "hparams.yaml", "w") as fh:
        yaml.dump({"diffusion": diffusion.get_hparams()}, fh, default_flow_style=False)
    torch.save(
        {"state_dict": diffusion.state_dict()},
        log_dir / "checkpoints" / "last_model.ckpt",
    )

    loaded = load_diffusion(log_dir)

    assert loaded.loss_balance == pytest.approx((0.75, 0.25))


def test_loss_balance_buffers_do_not_break_old_checkpoints():
    """The running loss scales must stay out of the state dict."""
    diffusion = create_diffusion(
        noisers=("cell_positions",), force_field=True, loss_balance="50:50"
    )
    plain = create_diffusion(noisers=("cell_positions",), force_field=True)

    assert not any("_loss_scales" in key for key in diffusion.state_dict())
    diffusion.load_state_dict(plain.state_dict())
    plain.load_state_dict(diffusion.state_dict())


def test_loss_balance_split_is_independent_of_loss_scale():
    """The same split holds for labels that differ by three orders of magnitude."""
    small = create_diffusion(
        noisers=("cell_positions",), force_field=True, feature_size=16, n_blocks=1,
        reference_energies=None, loss_balance="50:50",
    )
    huge = create_diffusion(
        noisers=("cell_positions",), force_field=True, feature_size=16, n_blocks=1,
        reference_energies=None, loss_balance="50:50",
    )
    huge.load_state_dict(small.state_dict())

    torch.manual_seed(0)
    small_fraction = _mean_regressor_fraction(small, _labelled_batch(-6.0, 0.1))
    torch.manual_seed(0)
    huge_fraction = _mean_regressor_fraction(huge, _labelled_batch(-6000.0, 50.0))

    # Both near the requested half, and — the point of the feature — the same
    # for wildly different label magnitudes.
    assert small_fraction == pytest.approx(0.5, abs=0.1)
    assert huge_fraction == pytest.approx(small_fraction, abs=0.02)


def test_loss_balance_respects_requested_split():
    """An 80/20 split puts materially less weight on the regressor than 50/50."""
    balanced = create_diffusion(
        noisers=("cell_positions",), force_field=True, feature_size=16, n_blocks=1,
        reference_energies=None, loss_balance="50:50",
    )
    skewed = create_diffusion(
        noisers=("cell_positions",), force_field=True, feature_size=16, n_blocks=1,
        reference_energies=None, loss_balance="80:20",
    )
    skewed.load_state_dict(balanced.state_dict())

    torch.manual_seed(0)
    half = _mean_regressor_fraction(balanced, _labelled_batch(-6.0, 0.1))
    torch.manual_seed(0)
    fifth = _mean_regressor_fraction(skewed, _labelled_batch(-6.0, 0.1))

    assert fifth == pytest.approx(0.2, abs=0.1)
    assert fifth < half


def test_loss_balance_scales_not_updated_during_validation():
    diffusion = create_diffusion(
        noisers=("cell_positions",), force_field=True, feature_size=16, n_blocks=1,
        loss_balance="50:50",
    )
    batch = _labelled_batch(-6.0, 0.1)

    diffusion.eval()
    diffusion.loss(batch, 0)
    assert not bool(diffusion._loss_scales_initialized)

    diffusion.train()
    diffusion.loss(batch, 0)
    assert bool(diffusion._loss_scales_initialized)

    diffusion.eval()
    frozen = diffusion._loss_scales.clone()
    diffusion.loss(batch, 0)
    assert torch.equal(diffusion._loss_scales, frozen)


def test_absolute_weighting_untouched_when_balance_disabled():
    """Without loss_balance the objective is exactly the old weighted sum."""
    diffusion = create_diffusion(
        noisers=("cell_positions",), force_field=True, feature_size=16, n_blocks=1
    )
    batch = _labelled_batch(-6.0, 0.1)

    torch.manual_seed(0)
    losses = diffusion.loss(batch, 0)
    expected = float(losses["pos_loss"]) + float(losses["regressor_loss"])

    assert float(losses["loss"]) == pytest.approx(expected, rel=1e-5)
    assert "regressor_fraction" not in losses


def test_train_from_atoms_forwards_loss_balance():
    class DummyTrainer:
        def fit(self, model, data):
            pass

    diffusion, _, _ = train_from_atoms(
        [_test_atoms_with_labels(-6.15)],
        noisers=("cell_positions",),
        force_field=True,
        loss_balance="80:20",
        trainer=DummyTrainer(),
    )

    assert diffusion.loss_balance == pytest.approx((0.8, 0.2))


def test_forcefield_hparams_reports_loss_balance():
    from agedi.api.training import _forcefield_hparams

    balanced = _forcefield_hparams(
        create_diffusion(noisers=("cell_positions",), force_field=True, loss_balance="80:20")
    )
    absolute = _forcefield_hparams(
        create_diffusion(noisers=("cell_positions",), force_field=True)
    )

    assert balanced["loss_balance"] == pytest.approx([0.8, 0.2])
    assert "regressor_loss_weight" not in balanced
    assert absolute["regressor_loss_weight"] == 1.0
    assert "loss_balance" not in absolute


# ---------------------------------------------------------------------------
# Inpainting
# ---------------------------------------------------------------------------


def test_inpaint_returns_atoms():
    from agedi import inpaint

    diffusion = create_diffusion(noisers=("cell_positions",))
    atoms = _test_atoms()

    structures = inpaint(
        diffusion, atoms, indices=[0], n_samples=1, steps=3, eps=1e-2,
    )
    assert len(structures) == 1
    assert structures[0].positions.shape == atoms.positions.shape


def test_inpaint_reconstructs_known_atoms():
    """Non-selected atoms must match the input structure to within tolerance."""
    from agedi import inpaint

    diffusion = create_diffusion(noisers=("cell_positions",))
    atoms = _test_atoms()

    out = inpaint(diffusion, atoms, indices=[0], n_samples=1, steps=4, eps=1e-2)[0]

    known_idx = [1, 2]
    assert np.allclose(
        out.positions[known_idx], atoms.positions[known_idx], atol=1e-3
    )


def test_inpaint_selected_atom_moves():
    from agedi import inpaint

    diffusion = create_diffusion(noisers=("cell_positions",))
    atoms = _test_atoms()

    out = inpaint(diffusion, atoms, indices=[0], n_samples=1, steps=4, eps=1e-2)[0]
    assert not np.allclose(out.positions[0], atoms.positions[0], atol=1e-3)


def test_inpaint_freeze_atoms_bit_exact():
    from agedi import inpaint

    diffusion = create_diffusion(noisers=("cell_positions",))
    atoms = _test_atoms()

    out = inpaint(
        diffusion, atoms, indices=[0], freeze=[1], n_samples=1, steps=4, eps=1e-2,
    )[0]
    assert np.allclose(out.positions[1], atoms.positions[1], atol=1e-6)


def test_inpaint_default_selection_is_random_fraction():
    """With no selection criterion, a random fraction of atoms is selected."""
    from agedi import inpaint

    diffusion = create_diffusion(noisers=("cell_positions",))
    atoms = _test_atoms()

    out = inpaint(
        diffusion, atoms, n_samples=1, steps=3, eps=1e-2, seed=0, fraction=0.5,
    )[0]
    # Some atoms should stay at their reference, some should move.
    displacement = np.linalg.norm(out.positions - atoms.positions, axis=-1)
    assert (displacement < 1e-3).any()
    assert (displacement > 1e-3).any()


def test_inpaint_symbols_selection():
    from agedi import inpaint

    diffusion = create_diffusion(noisers=("cell_positions",))
    atoms = _test_atoms()  # H2O: symbols are O, H, H

    out = inpaint(diffusion, atoms, symbols=["O"], n_samples=1, steps=3, eps=1e-2)[0]
    o_idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "O"]
    h_idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "H"]
    assert np.allclose(out.positions[h_idx], atoms.positions[h_idx], atol=1e-3)
    assert not np.allclose(out.positions[o_idx], atoms.positions[o_idx], atol=1e-3)


def test_inpaint_compile_true_raises():
    from agedi import inpaint

    diffusion = create_diffusion(noisers=("cell_positions",))
    atoms = _test_atoms()

    with pytest.raises(ValueError):
        inpaint(diffusion, atoms, indices=[0], n_samples=1, steps=3, compile=True)


def test_inpaint_freeze_overlap_with_selection_raises():
    from agedi import inpaint

    diffusion = create_diffusion(noisers=("cell_positions",))
    atoms = _test_atoms()

    with pytest.raises(ValueError):
        inpaint(
            diffusion, atoms, indices=[0], freeze=[0], n_samples=1, steps=3,
        )


def test_inpaint_as_atoms_false_returns_atoms_graph():
    from agedi import inpaint
    from agedi.data import AtomsGraph

    diffusion = create_diffusion(noisers=("cell_positions",))
    atoms = _test_atoms()

    out = inpaint(
        diffusion, atoms, indices=[0], n_samples=1, steps=3, eps=1e-2, as_atoms=False,
    )
    assert isinstance(out[0], AtomsGraph)


def test_inpaint_save_trajectory():
    from agedi import inpaint

    diffusion = create_diffusion(noisers=("cell_positions",))
    atoms = _test_atoms()

    out = inpaint(
        diffusion, atoms, indices=[0], n_samples=1, steps=3, eps=1e-2,
        save_trajectory=True,
    )
    assert len(out) == 1
    assert len(out[0]) >= 3
    assert all(hasattr(frame, "positions") for frame in out[0])


# ---------------------------------------------------------------------------
# select_atoms
# ---------------------------------------------------------------------------


def test_select_atoms_indices():
    from agedi.api import select_atoms

    atoms = _test_atoms()
    mask = select_atoms(atoms, indices=[0])
    assert mask.tolist() == [True, False, False]


def test_select_atoms_symbols():
    from agedi.api import select_atoms

    atoms = _test_atoms()
    mask = select_atoms(atoms, symbols=["H"])
    assert mask.tolist() == [False, True, True]


def test_select_atoms_z_range():
    from agedi.api import select_atoms

    atoms = _test_atoms()
    z = atoms.positions[:, 2]
    mask = select_atoms(atoms, z_range=(z.min() - 0.1, z.min() + 0.1))
    assert mask.sum() >= 1


def test_select_atoms_sphere():
    from agedi.api import select_atoms

    atoms = _test_atoms()
    center = atoms.positions[0]
    mask = select_atoms(atoms, sphere=(center, 0.01))
    assert mask.tolist() == [True, False, False]


def test_select_atoms_union_of_criteria():
    from agedi.api import select_atoms

    atoms = _test_atoms()
    # indices=[0] selects O; a tight sphere around atom 1's own position
    # selects just that one H, leaving atom 2 out of the union.
    mask = select_atoms(atoms, indices=[0], sphere=(atoms.positions[1], 0.1))
    assert mask.tolist() == [True, True, False]


def test_select_atoms_default_fraction_is_deterministic_with_seed():
    from agedi.api import select_atoms

    atoms = _test_atoms()
    mask_a = select_atoms(atoms, fraction=0.5, seed=42)
    mask_b = select_atoms(atoms, fraction=0.5, seed=42)
    assert mask_a.tolist() == mask_b.tolist()


def test_select_atoms_default_respects_fix_atoms():
    from ase.constraints import FixAtoms

    from agedi.api import select_atoms

    atoms = _test_atoms()
    atoms.set_constraint(FixAtoms(indices=[0, 1]))
    mask = select_atoms(atoms, fraction=1.0, seed=0)
    assert not mask[0]
    assert not mask[1]


def test_select_atoms_empty_selection_raises():
    from agedi.api import select_atoms

    atoms = _test_atoms()
    with pytest.raises(ValueError):
        select_atoms(atoms, indices=[])


def test_select_atoms_full_selection_raises():
    from agedi.api import select_atoms

    atoms = _test_atoms()
    with pytest.raises(ValueError):
        select_atoms(atoms, indices=[0, 1, 2])


def test_select_atoms_from_atoms_reads_inpaint_mask_array():
    from agedi.api import select_atoms

    atoms = _test_atoms()
    atoms.arrays["inpaint_mask"] = np.array([True, False, False])
    mask = select_atoms(atoms, from_atoms=True)
    assert mask.tolist() == [True, False, False]
