import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import pickle
from run_mortality_prediction import load_processed_data, stratified_split
import argparse
from keras.model import Model
from keras.layer import Input, LSTM, RepeatVector
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from numpy.random import seed
import numpy as np
from sklearn.mixture import GaussianMixture
import constants

def get_args():
    # Arguments that can be set by user
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clusters", type=int, default=3, help="Number of clusters for GMM. Type: int. Default = 3.")
    parser.add_argument("--gmm_tol", type=float, default=0.0001, help="Convergence threshold for GMM. Type: float. Default: 0.0001.")
    parser.add_argument("--data_hours", type=int, default=24, help="Number of hours from patient's stay to train on. Type: int. Default: 24.")
    parser.add_argument("--ae_epochs", type=int, default=100, help="Number of epochs. Type: int. Default: 100.")
    parser.add_argument("--ae_learning_rate", type=float, default=0.0001, help="Learning rate. Type: float. Default: 0.0001.")
    args = parser.parse_args()
    return args

def create_ae(X_train, X_val, learning_rate, latent_dim):
    """
    Create Autoencoder Model

    args:
        X_train: training data (shape = number of samples x number of timesteps x number of features)
        X_val: validation data 
        learning_rate: training learning rate
        latent_dim: hidden dimension
    returns:
        encoder
        sequence_autoencoder
    """
    num_timesteps = X_train.shape[1]
    num_features = X_train.shape[2]

    inputs = Input(shape=(num_timesteps, num_features))

    # Create Long Short-Term Memory Layer
    lstm1 = LSTM(latent_dim)     # Set number of nodes in hidden layer
    encoded = lstm1(inputs)

    # Create Repeat Vector that repeats the input n times
    decoded = RepeatVector(num_timesteps)(encoded)

    # Create Long Short-Term Memory Layer
    lstm2 = LSTM(num_features, return_sequences=True)
    decoded = lstm2(decoded)

    # Create sequence autoencoder and encoder models
    sequence_ae = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    sequence_ae.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    return encoder, sequence_ae

if __name__ == "__main__":
    # Get arguments inputted by user
    args = get_args()

    # Load data
    print("Loading Data...")
    X, Y, cohorts, saps_quartile, subject_ids = load_processed_data(args.data_hours, constants.GAP_TIME)
    Y = Y.astype(int)
    print("Done Loading Data.")

    # Get training, validation, and testing data
    X_train, X_val, X_test, Y_train, Y_val, Y_test, cohorts_train, cohorts_val, cohorts_test = \
        stratified_split(X, Y, cohorts)

    # Train autoencoder
    print("Training Autoencoder...")
    encoder, sequence_ae = create_ae(X_train, X_val, args.ae_learning_rate, constants.LATENT_DIM)
    early_stop = EarlyStopping(monitor='val_loss', patience=3)

    sequence_ae.fit(X_train, X_train, epochs=constants.ae_epochs, batch_size=constants.BATCH_SIZE, \
        shuffle=True, callbacks=[early_stop], validation_data=(X_val, X_val))

    if not os.path.exists('clustering_models/'):
        os.makedirs('clustering_models/')

    encoder.save("clustering_models/encoder_" + str(args.data_hours))
    sequence_ae.save("clustering_models/seq_ae_" + str(args.data_hours))
    print("Done Training Autoencoder.")

    # Encoder predictions
    encoder_predict_train = encoder.predict(X_train)
    encoder_predict_all = encoder.predict(X)

    # Train GMM
    print("Fit GMM...")
    gm = GaussianMixture(n_components=args.num_clusters, tol=args.gmm_tol)
    gm.fit(encoder_predict_train)
    pickle.dump(gm, open('clustering_models/gmm_' + str(args.data_hours), 'wb'))
    print("Done Fitting GMM.")

    cluster_pred = gm.predict(encoder_predict_all)

    if not os.path.exists('cluster_membership/'):
        os.makedirs('cluster_membership/')
    np.save('cluster_membership/' + 'test_clusters.npy', cluster_pred)




    
