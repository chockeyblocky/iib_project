"""
Original source: 2023 Qualcomm Technologies, Inc.
This has been adapted to use TensorFlow for academic purposes.
"""

from pathlib import Path

import numpy as np

random_seed = 0

from simulator import NBodySimulator


def generate_dataset(filename, simulator, num_samples, num_planets=5, domain_shift=False):
    """Samples from n-body simulator and stores the results at `filename`."""
    assert not Path(filename).exists()
    m, x_initial, v_initial, x_final, trajectories = simulator.sample(
        num_samples, num_planets=num_planets, domain_shift=domain_shift
    )
    np.savez(
        filename,
        m=m,
        x_initial=x_initial,
        v_initial=v_initial,
        x_final=x_final,
        trajectories=trajectories,
    )


def generate_datasets(path):
    """Generates a canonical set of datasets for the n-body problem, stores them in `path`."""
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)

    print(f"Creating gravity dataset in {str(path)}")

    simulator = NBodySimulator()
    generate_dataset(path / "train.npz", simulator, 100000, num_planets=3, domain_shift=False)
    generate_dataset(path / "val.npz", simulator, 5000, num_planets=3, domain_shift=False)
    generate_dataset(path / "eval.npz", simulator, 5000, num_planets=3, domain_shift=False)
    generate_dataset(
        path / "e3_generalization.npz", simulator, 5000, num_planets=3, domain_shift=True
    )
    generate_dataset(
        path / "object_generalization.npz", simulator, 5000, num_planets=5, domain_shift=False
    )

    print("Done, have a nice day!")


def main(data_dir="C:/Users/Christian/Documents/Coursework/iib_project/ga_transformer/datasets"):
    """Entry point for n-body dataset generation."""
    np.random.seed(random_seed)
    generate_datasets(data_dir)


if __name__ == "__main__":
    main()
