"""
This file runs the protein structure prediction experiment using a graph transformer and n generalised GA transformer
blocks.
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from proteins_model import cga_transformer_model, mlp_model

import pickle
import os

# set random seed
tf.random.set_seed(0)

# set path to data
PATH = "C:/Users/Christian/Documents/Coursework/iib_project/ga_transformer/"
DATA_PATH = PATH + "datasets/psp_dataset"


def get_nodes_and_edges(pdb, all_feat_paths, node_n, edge_n, dist_path):
    """
    Gets nodes and edges for input into graph transformer
    :param pdb: name of file to load
    :param all_feat_paths: all possible paths at which features lie
    :param node_n: number of features per node in GT input
    :param edge_n: number of layers of edges per node in each GT input
    :param dist_path: path to distance maps
    :return: nodes, edges, mask, length of feature
    """
    features = None
    for path in all_feat_paths:
        if os.path.exists(path + pdb + '.pkl'):
            features = pickle.load(open(path + pdb + '.pkl', 'rb'))

    l = len(features['seq'])

    nodes = np.zeros((1, l, node_n))
    edges = np.zeros((1, l, l, edge_n))
    mask = tf.cast(tf.ones((1, l)), tf.bool)

    # NODE FEATURES

    # Add secondary structure
    ss = features['ss']
    assert ss.shape == (3, l)
    fi = 0
    gi = 0

    for j in range(3):
        nodes[:, :, fi] = ss[j]
        fi += 1

    # Add PSSM
    pssm = features['pssm']
    assert pssm.shape == (l, 22)
    for j in range(22):
        nodes[:, :, fi] = pssm[:, j]
        fi += 1

    # Add SA
    sa = features['sa']
    assert sa.shape == (l,)
    nodes[:, :, fi] = sa
    fi += 1

    # Add entropy
    entropy = features['entropy']
    assert entropy.shape == (l,)
    nodes[:, :, fi] = entropy
    fi += 1

    # EDGE FEATURES

    # Add CCMpred
    ccmpred = features['ccmpred']
    assert ccmpred.shape == ((l, l))
    edges[:, :, :, gi] = ccmpred
    gi += 1
    # Add FreeContact
    freecon = features['freecon']
    assert freecon.shape == ((l, l))
    edges[:, :, :, gi] = freecon
    gi += 1
    # Add potential
    potential = features['potential']
    assert potential.shape == ((l, l))
    edges[:, :, :, gi] = potential
    gi += 1
    # Add distance map
    distance = np.load(dist_path + pdb + '-ca.npy', allow_pickle=True)
    edges[:, :, :, gi] = distance

    # if tensors are required in batch generation then use this
    nodes = tf.convert_to_tensor(nodes)
    edges = tf.convert_to_tensor(edges)

    return nodes, edges, mask, l


class EarlyStopper:
    """
    Early stopper class used in training.
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def custom_loss(pred_dist, Y, alpha, batch_size, mae):
    """
    Custom loss which avoids gradient instability
    :param pred_dist: predicted distance map
    :param Y: actual distance map
    :param alpha: weighting of ssim loss
    :param batch_size: batch size
    :param mae: instance of tf.keras.MeanAbsoluteError class
    :return:
    """
    loss1 = tf.cast(tf.reduce_mean(mae(pred_dist, Y)), tf.float32)
    loss2 = tf.cast(1 - tf.image.ssim(img1=pred_dist, img2=Y, max_val=255), tf.float32)
    return (loss1 + alpha * loss2) / batch_size


def custom_norm(x):
    """
    Custom norm required to not return nan gradients.
    """
    # added constant required to avoid nans
    return tf.math.sqrt(tf.reduce_sum(tf.square(x), axis=-1) + 1.0e-12)


def save_model(name, model):
    """
    Saves model weights as .pkl files
    :param name: String under which to save model
    :param model: TensorFlow model to save
    :return:
    """
    with open(PATH + 'models/{}.pkl'.format(name), 'wb') as f:
        pickle.dump(model.get_weights(), f)


def orient_coords(x, y):
    """
    Computes x coordinates oriented such that MSE to y is minimised (assumes centroids of both x and y have already been
    normalised to 0). Follows algorithm found in "Least-Squares Fitting of Two 3-D Point Sets" (Arun et al.).
    :param x: tf tensor containing (1, N, 3) predicted points
    :param y: tf tensor containing (1, N, 3) target points
    :return: x_oriented
    """
    # compute matrix from outer product of coordinate pairs
    H = tf.reshape(tf.einsum("...ki,...kj->...ij", x, y), (3, 3))

    # compute svd of H
    s, u, v = tf.linalg.svd(H)

    # compute rotation matrix (R = vu^t)
    R = tf.einsum("ik,jk->ij", v, u)

    # apply rotation matrix to x and return
    return tf.einsum("ik,...k->...i", R, x)




def main():
    """
    Main function to run.
    :return:
    """
    deepcov_features_path = DATA_PATH + '/data/deepcov/features/'
    deepcov_distances_path = DATA_PATH + '/data/deepcov/ca_distance/'
    deepcov_coords_path = DATA_PATH + '/data/deepcov/ca_coords/'
    psicov_features_path = DATA_PATH + '/data/psicov/features/'
    psicov_distances_path = DATA_PATH + '/data/psicov/ca_distance/'
    psicov_coords_path = DATA_PATH + '/data/psicov/ca_coords/'

    lst = os.listdir(deepcov_features_path) + os.listdir(psicov_features_path)
    lst.sort()

    lst_train = []
    for filename in lst:
        pdb = os.path.splitext(filename)[0]
        if os.path.exists(deepcov_features_path + pdb + '.pkl'):
            lst_train = np.append(lst_train, filename)

    # lst_test is separate from evaluation set used during training
    lst_test = []
    for filename in lst:
        pdb = os.path.splitext(filename)[0]
        if os.path.exists(psicov_features_path + pdb + '.pkl'):
            lst_test = np.append(lst_test, filename)

    node_n = 27
    edge_n = 4

    model = cga_transformer_model(num_blocks=2, num_edge_layers=edge_n, num_features=node_n)
    # model = mlp_model(num_edge_layers=edge_n, num_features=27)

    train_feat_paths = [deepcov_features_path]
    test_feat_paths = [psicov_features_path]

    initial_learning_rate = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.98,
        staircase=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # early_stopper = EarlyStopper(patience=15, min_delta=0.1)
    stopflag = 0

    lsttrain, lstval = train_test_split(lst_train, test_size=0.20, random_state=42)

    epochs = 100
    batch_size = 10
    n_batches = len(lsttrain) // batch_size

    batch_loss = 0
    total_loss = 0
    i_in = 0
    j_in = 0
    alpha = 20
    mse = tf.keras.losses.MeanSquaredError()

    final_loss = []
    val_loss_arr = []

    for i in range(i_in, epochs):
        print('****')
        print("epoch {}/{}".format(i, epochs))

        # reshuffle before each epoch
        np.random.shuffle(lsttrain)
        np.random.shuffle(lstval)

        for j in range(j_in, n_batches):

            print("batch n.", j + 1, "/", n_batches)
            if (j + 1) * batch_size < len(lsttrain):
                lstbatch = lsttrain[j * batch_size:(j + 1) * batch_size]
            else:
                lstbatch = lsttrain[j * batch_size:]

            nodes_batch = []
            edges_batch = []
            mask_batch = []
            coords_batch = []
            grads = []

            # collect data from batch
            for filename in lstbatch:
                # remove .pkl extension
                filename = os.path.splitext(filename)[0]

                # load nodes, edges, mask
                nodes, edges, mask, l = get_nodes_and_edges(filename, train_feat_paths, node_n, edge_n,
                                                            deepcov_distances_path)
                nodes_batch.append(nodes)
                edges_batch.append(edges)
                mask_batch.append(mask)

                # load true coordinates - for loss calculation
                coords_batch.append(tf.convert_to_tensor(np.load(deepcov_coords_path + filename + ".npy",
                                                                 allow_pickle=True), dtype=tf.float32))

            for i in range(batch_size):
                with tf.GradientTape() as tape:
                    # watch batched tensors
                    tape.watch(nodes_batch[i])
                    tape.watch(edges_batch[i])

                    # feed forward into model
                    predicted_coords = model(nodes_batch[i], edges_batch[i], mask=mask_batch[i])

                    # orient predicted coords with respect to actual ones
                    oriented_coords = orient_coords(predicted_coords, coords_batch[i])

                    # calculate MSE loss
                    loss = mse(coords_batch[i], oriented_coords) / batch_size

                # record batch loss for output
                batch_loss += loss

                # update gradients
                if len(grads) == 0:
                    grads = tape.gradient(loss, model.trainable_weights)
                else:
                    new_grads = tape.gradient(loss, model.trainable_weights)
                    for i in range(len(grads)):
                        grads[i] = grads[i] + new_grads[i]

            del nodes_batch, edges_batch, mask_batch, coords_batch, predicted_coords, oriented_coords

            # print("Loss: ", batch_loss)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            total_loss += batch_loss
            batch_loss = 0

        print("Validation")
        tot_val_loss = 0
        CNT = 0

        for filename in lstval:
            # print(filename)
            CNT += 1
            print(CNT, "/", len(lstval))

            filename = os.path.splitext(filename)[0]

            nodes, edges, mask, l = get_nodes_and_edges(filename, train_feat_paths, node_n, edge_n,
                                                        deepcov_distances_path)
            actual_coords = tf.convert_to_tensor(np.load(deepcov_coords_path + filename + ".npy",
                                                         allow_pickle=True), dtype=tf.float32)

            # feed forward into model
            predicted_coords = model(nodes, edges, mask=mask)

            # orient predicted coords with respect to actual ones
            oriented_coords = orient_coords(predicted_coords, actual_coords)

            # calculate MSE loss
            tot_val_loss += mse(actual_coords, oriented_coords) / batch_size

            del nodes, edges, mask, l, oriented_coords, actual_coords, predicted_coords

        val_loss_arr = np.append(val_loss_arr, (tot_val_loss * batch_size / len(lstval)))
        final_loss = np.append(final_loss, total_loss * batch_size / len(lsttrain))

        print("....")
        print("validation loss:", val_loss_arr)
        print("training loss:", final_loss)
        print("....")

        # if early_stopper.early_stop(tot_val_loss / len(lstval)):
        #     stopflag = 1
        #     break

        tot_val_loss = 0
        total_loss = 0

        if j_in != 0:
            j_in = 0

        if stopflag == 1:
            break

    save_model("psp_cgatr_2_block_new_schedule", model)


if __name__ == "__main__":
    main()
