
def get_dataset(
    atoms,
    neighbour_list,
    path="dataset.db",
    repeats=[1, 2],
    mask_below_h=2.6,
    z_confinement=[2.5, 7.8],
    batch_size=32,
):
    import os
    import random

    import numpy as np
    import schnetpack.transform as trn
    from ase.calculators.singlepoint import SinglePointCalculator
    from schnetpack.data import ASEAtomsData, AtomsDataModule

    data = []
    for a in atoms:
        e = a.get_potential_energy()
        f = a.get_forces(apply_constraint=False).reshape(-1, 3)
        for r in repeats:
            a1 = a.copy()
            a1 = a1.repeat([r, r, 1])
            f1 = np.vstack([f] * r**2)
            a1.set_calculator(SinglePointCalculator(a1, energy=r**2 * e, forces=f1))

            data.append(a1)

    atoms = data
    

    print("=" * 10, "Creating dataset", "=" * 10)
    if os.path.exists(path):
        os.remove(path)
        if os.path.exists("split.npz"):
            os.remove("split.npz")

    property_list = []
    for a in atoms:
        e = a.get_potential_energy()
        f = a.get_forces().reshape(-1, 3)
        c = SinglePointCalculator(a, energy=e, forces=f)
        a.set_calculator(c)
        mask = np.zeros_like(f, dtype=bool)
        mask[a.get_positions()[:, 2] < mask_below_h] = True

        properties = {
            "energy": np.array([e]),
            "forces": f,
            "mask": mask,
            "z_confinement": z_confinement,
        }
        property_list.append(properties)

    dataset = ASEAtomsData.create(
        path,
        distance_unit="Ang",
        property_unit_dict={
            "energy": "eV",
            "forces": "eV/Ang",
            "mask": None,
            "z_confinement": None,
        },
    )
    dataset.add_systems(property_list, atoms)

    print("Number of reference calculations:", len(dataset))
    print("Available properties:")

    for p in dataset.available_properties:
        print("-", p)

    example = dataset[0]
    print("Properties of item in dataset:")

    for k, v in example.items():
        print("-", k, ":", v.shape)

    print("=" * 10, "Finished dataset", "=" * 10)

    dataset = AtomsDataModule(
        path,
        batch_size=batch_size,
        num_train=0.90,
        num_val=0.1,
        transforms=[
            neighbour_list,
            trn.CastTo32(),
        ],
        num_workers=32,
        pin_memory=True,
        split_file="split.npz",
    )
    dataset.prepare_data()
    dataset.setup()

    train_dataloader = dataset.train_dataloader()
    example = next(iter(train_dataloader))

    print("Properties of batch:")

    for k, v in example.items():
        print("-", k, ":", v.shape)

    print("idx:", example["_idx_m"])

    return dataset


def get_diffusion_model(
    cutoff=6.0,
    n_atom_basis=64,
    n_rbf=30,
    n_interactions=4,
    gated_blocks=4,
    beta_max=3.0,
    beta_min=1e-2,
    lr=1e-3,
    neighbour_list=None,
):
    import schnetpack as spk

    from agedi.diffusion import VPDiffusion
    from agedi.models import ConditionedScoreModel, Potential
    from agedi.utils import TorchNeighborList

    if neighbour_list is None:
        neighbour_list = TorchNeighborList(cutoff)

    radial_basis = spk.nn.GaussianRBF(n_rbf=n_rbf, cutoff=cutoff)
    representation = spk.representation.PaiNN(
        n_atom_basis=n_atom_basis,
        n_interactions=n_interactions,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )

    score_model = ConditionedScoreModel(
        representation, time_dim=2, gated_blocks=gated_blocks
    )

    pred_energy = spk.atomistic.Atomwise(
        n_in=representation.n_atom_basis, output_key="energy"
    )
    pred_forces = spk.atomistic.Forces(energy_key="energy", force_key="forces")
    pairwise_distance = spk.atomistic.PairwiseDistances()
    potential = Potential(
        representation=representation,
        input_modules=[pairwise_distance],
        output_modules=[pred_energy, pred_forces],
    )

    diffusion = VPDiffusion(
        score_model=score_model,
        potential_model=potential,
        neighbour_list=neighbour_list,
        beta_max=beta_max,
        beta_min=beta_min,
        optim_config={"lr": lr},
        scheduler_config={"factor": 0.90, "patience": 100},
    )

    return diffusion, neighbour_list


def sample(
        diffusion, num_samples, template, symbols, z_confinement, num_steps=1000, eta=1e-2, postrelax_steps=100,
        return_trajectories=False,
):
    import numpy as np
    import schnetpack as spk
    import torch
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    def to_atoms(batch_list):
        atoms = []
        for b in batch_list:
            a = Atoms(
                numbers=b["_atomic_numbers"].cpu().detach().numpy(),
                positions=b["_positions"].cpu().detach().numpy(),
                cell=b["_cell"].cpu().detach().numpy().reshape(3, 3),
                pbc=b["_pbc"].cpu().detach().numpy(),
            )
            try:
                e, f = b["energy"].cpu().item(), b["forces"].cpu().detach().numpy().reshape(-1, 3)
                a.set_calculator(SinglePointCalculator(a, energy=e, forces=f))                
            except:
                print('No predicted energies and forces')

            atoms.append(a)
        return atoms

    converter = spk.interfaces.AtomsConverter(
        neighbor_list=None,
        additional_inputs={
            "mask": torch.tensor(
                np.vstack(
                    [
                        np.ones((len(template), 3), dtype=bool),
                        np.zeros((len(symbols), 3), dtype=bool),
                    ]
                )
            ),
            "z_confinement": z_confinement,
        },
        device='cuda'
    )

    # generate data
    n_split = 64 if num_samples > 64 else num_samples

    all_atoms = []
    for i in range(num_samples // n_split):
        atoms_data = []
        for _ in range(n_split):
            all_symbols = template.get_chemical_symbols() + symbols #["Ag"] * len(template) + symbols
            positions = np.vstack(
                (template.get_positions(), np.zeros((len(symbols), 3)))
            )
            atoms_data.append(
                Atoms(
                    all_symbols,
                    positions=positions,
                    cell=template.get_cell(),
                    pbc=template.get_pbc(),
                )
            )

        data = converter(atoms_data)
        data["_pbc"] = data["_pbc"].view(-1)  # hack

        if return_trajectories:
            batch, traj_batch = diffusion.regressor_guidance_sample(
                data, num_steps=num_steps, save_traj=True, eta=eta, postrelax_steps=postrelax_steps,
            )

            # #save traj
            all_trajs = [[] for _ in range(n_split)]
            atoms_trajs = [[] for _ in range(n_split)]
            for b in traj_batch:
                batch_list = diffusion._split_batch(b)
                for j, item in enumerate(batch_list):
                    all_trajs[j].append(item)

            for j, batch_list in enumerate(all_trajs):
                atoms = to_atoms(batch_list)
                atoms_trajs[j] = atoms
        else:
            batch = diffusion.regressor_guidance_sample(
                data, num_steps=num_steps, save_traj=False, eta=eta, postrelax_steps=postrelax_steps
            )

        # save final
        batch_list = diffusion._split_batch(batch, keep_ef=True)
        atoms = to_atoms(batch_list)
        all_atoms += atoms

    if return_trajectories:
        return all_atoms, atoms_trajs
    else:
        return all_atoms
