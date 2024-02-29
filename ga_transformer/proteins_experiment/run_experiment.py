"""
This file runs the protein structure prediction experiment using a graph transformer and n generalised GA transformer
blocks.
"""

import tensorflow as tf
import numpy as np
from tfga import GeometricAlgebra
from sklearn.model_selection import train_test_split
import time
from layers.graph_transformer import GraphTransformer
from proteins_model import cga_transformer_model

import pickle
import os

# set random seed
tf.random.set_seed(0)

# set path to data
PATH = "C:/Users/Christian/Documents/Coursework/iib_project/ga_transformer/"
DATA_PATH = PATH + "datasets/psp_dataset"


def get_nodes_and_edges(pdb, all_feat_paths, node_n, edge_n):
    """
    Gets nodes and edges for input into graph transformer
    :param pdb: name of file to load
    :param all_feat_paths: all possible paths at which features lie
    :param node_n: number of features per node in GT input
    :param edge_n: number of layers of edges per node in each GT input
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


class Net(tf.keras.Model):
    def __init__(self, edge_n):
        super().__init__()
        self.gt = GraphTransformer(
            depth=3,
            heads=4,
            edge_dim=edge_n,
            with_feedforwards=True,
            rel_pos_emb=True)
        self.dense = tf.keras.layers.Dense(3, activation=None)
        # add in shape reduction layers to make shapes match

    def call(self, nodes, edges, mask):
        x, edges_new = self.gt(nodes, edges, mask=mask)
        x = self.dense(x)
        x = tf.reshape(x, (1, -1, 3))
        return x


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


def main():
    """
    Main function to run.
    :return:
    """
    deepcov_features_path = DATA_PATH + '/data/deepcov/features/'
    deepcov_distances_path = DATA_PATH + '/data/deepcov/distance/'
    psicov_features_path = DATA_PATH + '/data/psicov/features/'
    psicov_distances_path = DATA_PATH + '/data/psicov/distance/'

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
    edge_n = 3

    model = cga_transformer_model(num_blocks=1, num_edge_layers=edge_n, num_features=node_n)
    save_model("test", model)

    train_feat_paths = [deepcov_features_path]
    test_feat_paths = [psicov_features_path]

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    early_stopper = EarlyStopper(patience=4, min_delta=0.1)
    stopflag = 0

    lsttrain, lstval = train_test_split(lst_train, test_size=0.20, random_state=42)

    epochs = 1  # down from 100 for testing purposes
    batch_size = 10
    n_batches = len(lsttrain) // batch_size

    batch_loss = 0
    total_loss = 0
    i_in = 0
    j_in = 0
    alpha = 20
    mae = tf.keras.losses.MeanAbsoluteError()

    final_loss = []
    val_loss_arr = []

    for i in range(i_in, epochs):
        print('****')

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
            dist_batch = []
            grads = []

            # collect data from batch
            for filename in lstbatch:
                # remove .pkl extension
                filename = os.path.splitext(filename)[0]

                # load nodes, edges, mask
                nodes, edges, mask, l = get_nodes_and_edges(filename, train_feat_paths, node_n, edge_n)
                nodes_batch.append(nodes)
                edges_batch.append(edges)
                mask_batch.append(mask)

                # load distance - for loss calculation
                distance = np.load(deepcov_distances_path + filename + '-cb.npy', allow_pickle=True)
                dist_batch.append(tf.convert_to_tensor(distance[2]))

            for i in range(batch_size):
                with tf.GradientTape() as tape:
                    # watch batched tensors
                    tape.watch(nodes_batch[i])
                    tape.watch(edges_batch[i])

                    # feed forward into model
                    coord = model(nodes_batch[i], edges_batch[i], mask=mask_batch[i])

                    # convert to predicted distances
                    coord = tf.repeat(coord, repeats=coord.shape[1], axis=0)
                    pred_dist = tf.expand_dims(
                        tf.expand_dims(custom_norm(coord - tf.transpose(coord, perm=[1, 0, 2])), 0), -1)
                    # could do a reshape instead

                    # get true distance
                    actual_dist = tf.expand_dims(tf.expand_dims(tf.cast(dist_batch[i], tf.float32), -1), 0)

                    # compute loss
                    loss = custom_loss(pred_dist, actual_dist, alpha, batch_size, mae)

                # record batch loss for output
                batch_loss += loss

                # update gradients
                if len(grads) == 0:
                    grads = tape.gradient(loss, model.trainable_weights)
                else:
                    new_grads = tape.gradient(loss, model.trainable_weights)
                    for i in range(len(grads)):
                        grads[i] = grads[i] + new_grads[i]

            del nodes_batch, edges_batch, mask_batch, dist_batch, coord, actual_dist, pred_dist

            print("Loss: ", batch_loss)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            start = time.time()
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

            nodes, edges, mask, l = get_nodes_and_edges(filename, train_feat_paths, node_n, edge_n)
            distance = tf.convert_to_tensor(np.load(deepcov_distances_path + filename + '-cb.npy', allow_pickle=True))

            # feed forward into model
            coord = model(nodes, edges, mask=mask)

            # convert to predicted distances
            coord = tf.repeat(coord, repeats=coord.shape[1], axis=0)
            pred_dist = tf.expand_dims(tf.expand_dims(custom_norm(coord - tf.transpose(coord, perm=[1, 0, 2])), 0),
                                       -1)  # could do a reshape instead

            # compute loss
            actual_dist = tf.expand_dims(tf.expand_dims(tf.cast(distance, tf.float32), -1), 0)

            loss = custom_loss(pred_dist, actual_dist, alpha, batch_size, mae)

            tot_val_loss += loss

            del nodes, edges, mask, l, coord, distance, actual_dist, pred_dist

        val_loss_arr = np.append(val_loss_arr, (tot_val_loss / len(lstval)))
        final_loss = np.append(final_loss, total_loss / len(lsttrain))

        print("....")
        print("validation loss:", val_loss_arr)
        print("training loss:", final_loss)
        print("....")

        if early_stopper.early_stop(tot_val_loss / len(lstval)):
            stopflag = 1
            break

        tot_val_loss = 0
        total_loss = 0

        if j_in != 0:
            j_in = 0

        if stopflag == 1:
            break


if __name__ == "__main__":
    main()
