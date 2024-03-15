"""
This file will generate the coordinates as .pkl objects from .ent files to be used in training.
"""
from Bio.PDB.PDBParser import PDBParser
import argparse
import pickle
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# set path to data
PATH = "C:/Users/Christian/Documents/Coursework/iib_project/ga_transformer/"
DATA_PATH = PATH + "datasets/psp_dataset"


def main():
    """
    Main function to run.
    :return:
    """
    coords_path = DATA_PATH + "/data/chains-test/"
    deepcov_features_path = DATA_PATH + '/data/deepcov/features/'
    deepcov_distances_path = DATA_PATH + '/data/deepcov/distance/'
    psicov_features_path = DATA_PATH + '/data/psicov/features/'
    psicov_distances_path = DATA_PATH + '/data/psicov/distance/'

    coord_save_path = DATA_PATH + '/data/psicov/ca_coords/'
    dist_save_path = DATA_PATH + '/data/psicov/ca_distance/'

    lst = os.listdir(coords_path)
    lst.sort()

    lst_train = os.listdir(deepcov_features_path)
    lst_test = os.listdir(psicov_features_path)

    parser = PDBParser(PERMISSIVE=1)
    structure_id = "chain"

    for filename in lst_test:
        filename = os.path.splitext(filename)[0]
        print(filename)

        if filename + '.pdb' in lst:
            structure = parser.get_structure(structure_id, coords_path + filename + '.pdb')

            N = 0
            idx = 0

            features = pickle.load(open(psicov_features_path + filename + '.pkl', 'rb'))
            print("PSICOV LENGTH: ", len(features['seq']))

            # distance = np.load(deepcov_distances_path + filename + '-cb.npy', allow_pickle=True)[2]
            # print(distance.shape)

            # counting the total number of atoms N in the chain
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            if atom.altloc == "B":
                                del atom

            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            N += 1
                            if atom.name == "CA":
                                idx += 1

            print("FOUND: ", idx)
            P = np.zeros([N, 3])
            cnt = np.zeros([idx, 1])

            i = 0
            m = 0
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            P[m] = atom.get_coord()
                            m += 1
                            if atom.name == "CA":
                                cnt[i, 0] = m
                                i += 1

            coord = []

            for m in range(0, idx):
                i = int(cnt[m, 0]) - 1
                coord = np.append(coord, P[i, :])

            coord = np.reshape(coord, [idx, 3])

            mean_x = np.mean(coord[:, 0])
            mean_y = np.mean(coord[:, 1])
            mean_z = np.mean(coord[:, 2])

            M = [mean_x, mean_y, mean_z]

            coord = coord - M

            # store coord
            new_dist = np.zeros((idx, idx))
            for i in range(len(coord)):
                for j in range(len(coord)):
                    new_dist[i, j] = np.sqrt((coord[i, 0] - coord[j, 0]) ** 2 + (coord[i, 1] - coord[j, 1]) ** 2 +
                                             (coord[i, 2] - coord[j, 2]) ** 2)

            np.save(coord_save_path + filename + '.npy', coord)
            np.save(dist_save_path + filename + '-ca.npy', new_dist)
        else:
            print("NOT FOUND")


if __name__ == "__main__":
    main()
