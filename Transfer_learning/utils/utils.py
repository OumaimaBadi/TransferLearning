import os
import sys
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model,Sequential
#from aeon.datasets import load_classification
from sklearn.preprocessing import LabelEncoder,StandardScaler
from tensorflow.keras.utils import to_categorical
from aeon.classification.sklearn import RotationForestClassifier
#import contextlib
from aeon.transformations.collection.feature_based import Catch22 



def build_mlp_model(input_dim, num_classes):
    """
    Build the MLP model for feature extraction and classification.

    Parameters:
    input_dim (int): Dimension of the input features.
    num_classes (int): Number of output classes.

    Returns:
    Model: Compiled MLP model.
    """

    mlp_input = Input(shape=(input_dim,))
    mlp = Dense(128, activation='relu')(mlp_input)
    mlp = Dense(64, activation='relu')(mlp)
    mlp = Dense(32, activation='relu')(mlp)
    mlp_output = Dense(num_classes, activation='softmax')(mlp)

    mlp_model = Model(inputs=mlp_input, outputs=mlp_output)

    mlp_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return mlp_model


def create_classifier(num_classes, input_dim,X_train,y_train_categorical):
    classifier = Sequential([
        Dense(num_classes, activation='softmax', input_shape=(input_dim,))
    ])
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.fit(X_train, y_train_categorical, epochs=1500, batch_size=64, verbose=1)
    return classifier


def fit_predict_rotation_forest(latent_space_train, y_train, latent_space_test):
    rf = RotationForestClassifier()
    rf.fit(latent_space_train, y_train)
    predictions = rf.predict_proba(latent_space_test)
    return predictions


def write_results(results, headers, filepath):
    """Writes the evaluation results to a CSV file."""
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for result in results:
            writer.writerow(result)



def load_features(dataset_name, features_directory):
    train_features_path = os.path.join(features_directory, dataset_name, 'features_train.csv')
    test_features_path = os.path.join(features_directory, dataset_name, 'features_test.csv')
    custom_features_train = pd.read_csv(train_features_path).values
    custom_features_test = pd.read_csv(test_features_path).values
    #print(custom_features_train)
    #print(custom_features_train.shape)
    return custom_features_train, custom_features_test



def load_and_prepare_data(dataset_name, features_directory):
    """Loads and preprocesses data for a given dataset."""
    preprocessor = TimeSeriesPreprocessor()
    try:
        X_train, y_train, X_test, y_test = preprocessor.preprocess(dataset_name)
        train_features, test_features = load_features(dataset_name, features_directory)

    except Exception as e:
        print(f"Error loading or preparing data for {dataset_name}: {e}")
        return None
    return X_train, y_train, X_test, y_test, train_features, test_features

def calculate_catch22(self, dataset_name, base_output_dir='/home/obadi/catche22_FineTuning/catch22_features'):
    # Prétraiter les données
    xtrain, _ , xtest, _ = self.preprocess(dataset_name)
    if xtrain is None or xtest is None:
        return

    # Initialiser Catch22 et StandardScaler
    tnf = Catch22(replace_nans=True)
    scaler = StandardScaler()

    # Calculer les caractéristiques Catch22 pour l'ensemble d'entraînement
    catch22_train = tnf.fit_transform(xtrain)
    catch22_train_scaled = scaler.fit_transform(catch22_train)

    # Calculer les caractéristiques Catch22 pour l'ensemble de test
    catch22_test = tnf.transform(xtest)
    catch22_test_scaled = scaler.transform(catch22_test)

    # Créer le répertoire de sortie spécifique à l'ensemble de données s'il n'existe pas
    output_dir = os.path.join(base_output_dir, dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Définir les noms des colonnes pour Catch22
    catch22_feature_names = [
        "DN_HistogramMode_5", "DN_HistogramMode_10", "SB_BinaryStats_mean_longstretch1",
        "DN_OutlierInclude_p_001_mdrmd", "DN_OutlierInclude_n_001_mdrmd", "CO_f1ecac",
        "CO_FirstMin_ac", "CO_HistogramAMI_even_2_5", "IN_AutoMutualInfoStats_40_gaussian_fmmi",
        "MD_hrv_classic_pnn40", "SB_BinaryStats_diff_longstretch0", "SB_MotifThree_quantile_hh",
        "FC_LocalSimple_mean1_tauresrat", "CO_Embed2_Dist_tau_d_expfit_meandiff", "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
        "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", "SP_Summaries_welch_rect_area_5_1",
        "SP_Summaries_welch_rect_centroid", "FC_LocalSimple_mean3_stderr", "CO_trev_1_num",
        "PD_PeriodicityWang_th0_01", "CO_Embed2_Dist_tau_d_expfit_meandiff"
    ]

    # Convertir les caractéristiques en DataFrames
    df_train = pd.DataFrame(catch22_train_scaled, columns=catch22_feature_names)
    df_test = pd.DataFrame(catch22_test_scaled, columns=catch22_feature_names)

    # Sauvegarder les DataFrames en fichiers CSV dans le répertoire spécifique à l'ensemble de données
    train_csv_path = os.path.join(output_dir, 'features_train.csv')
    test_csv_path = os.path.join(output_dir, 'features_test.csv')

    df_train.to_csv(train_csv_path, index=False)
    df_test.to_csv(test_csv_path, index=False)

    print(f"Caractéristiques Catch22 sauvegardées pour {dataset_name}:")
    print(f" - Entraînement : {train_csv_path}")
    print(f" - Test : {test_csv_path}")




# def load_models_2(models_directory, dataset_name, multimodal=False):
    
#     if multimodal:
#         model_paths = [os.path.join(models_directory, dataset_name, f"{dataset_name}_best_multimodal_{i}.hdf5") for i in range(5)]
#     else:
#         model_paths = [os.path.join(models_directory, dataset_name, f"{dataset_name}_best_model_{i}.hdf5") for i in range(5)]
    
#     models = []
#     for path in model_paths:
#         try:
#             model = tf.keras.models.load_model(path)
#             models.append(model)
#         except Exception as e:
#             print(f"Failed to load model from {path}: {e}")
#     return models

def load_models(models_directory, dataset_name):

    models = []
    for run in range(5): 
        run_path = os.path.join(models_directory, f"run_{run}", dataset_name, "best_model.hdf5")
        try:
            model = tf.keras.models.load_model(run_path)
            models.append(model)
        except Exception as e:
            print(f"Failed to load model from {run_path}: {e}")

    return models

def load_latent_models(models_directory, dataset_name):
    
    models = load_models(models_directory, dataset_name)
    latent_models = []
    for model in models:
        try:
            latent_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
            latent_models.append(latent_model)
        except Exception as e:
            print(f"Error creating latent model: {e}")
    return latent_models



def evaluate_models(dataset_name, models_directory, features_directory,evaluation_type,multimodal=False):
        """
        Evaluate multimodal models using the test dataset.

        Parameters:
        dataset_name (str): Name of the dataset.

        Returns:
        Evaluation results.
        """
        def calculate_mean_accuracy(models, X_test, y_test):
            accuracies = [model.evaluate(X_test, y_test, verbose=0)[1] for model in models]
            print(len(accuracies))
            return sum(accuracies) / len(accuracies)
        
        data = load_and_prepare_data(dataset_name, features_directory)
        if data is None:
            return None

        _, _, X_test, y_test, _, test_features = data
        models = load_models(models_directory, dataset_name, multimodal)
        num_classes = len(np.unique(y_test))
        y_test_one_hot = to_categorical(y_test, num_classes=num_classes)


        if evaluation_type == 'lite':
            mean_accuracy = calculate_mean_accuracy(models, [X_test,test_features], y_test_one_hot)
            return [dataset_name,mean_accuracy]
        else :

            probabilities = [model.predict([X_test,test_features]) for model in models]
            averaged_probabilities = combine_probabilities(probabilities)
        
        return evaluate_and_record_results(y_test, averaged_probabilities, dataset_name)


def evaluate_rotation_forest_ensembling(dataset_name, models_directory, features_directory,multimodal=False):
        """
        Evaluate multimodal models using the latent space of the multimodal model then feed those features to the RotationForest Classifier.

        Parameters:
        dataset_name (str): Name of the dataset.

        Returns:
        Evaluation results.
        """
        data = load_and_prepare_data(dataset_name, features_directory)
        if data is None:
            return None

        X_train, y_train, X_test, y_test, train_features, test_features = data
        latent_models = load_latent_models(models_directory, dataset_name,multimodal=False)
        combined_probabilities = []

        for latent_model in latent_models:
            latent_space_train = latent_model.predict([X_train, train_features])
            latent_space_test = latent_model.predict([X_test, test_features])

            predictions=fit_predict_rotation_forest(latent_space_train, y_train, latent_space_test)
            combined_probabilities.append(predictions)

        final_probabilities = combine_probabilities(combined_probabilities)
        return evaluate_and_record_results(y_test, final_probabilities, dataset_name)


def concatenate_features(feature_sets):
    return np.concatenate(feature_sets, axis=1)




def combine_probabilities(probabilities):
    """Combines model probabilities by averaging them."""
    mean_probabilities = np.mean(probabilities, axis=0)
    return np.argmax(mean_probabilities, axis=1)




def evaluate_and_record_results(y_true, predictions, dataset_name):
    """Evaluates model predictions and prepares results for recording."""
    acc = accuracy_score(y_true, predictions)
    f1 = f1_score(y_true, predictions, average='weighted')
    precision = precision_score(y_true, predictions, average='weighted')
    recall = recall_score(y_true, predictions, average='weighted')
    return [dataset_name, acc, f1, precision, recall]




class TimeSeriesPreprocessor:
    def __init__(self):
        self.encoder = LabelEncoder()

    def load_data(self,dataset_name):

        folder_path = "/home/oumaima/Transfer_learning/datasets/UCRArchive_2018/"
        folder_path += dataset_name  + "/"

        train_path = folder_path + dataset_name  + "_TRAIN.tsv"
        test_path = folder_path + dataset_name  + "_TEST.tsv"

        if os.path.exists(test_path) <= 0:
            print("File not found")
            return None, None, None, None

        train = np.loadtxt(train_path, dtype=np.float64)
        test = np.loadtxt(test_path, dtype=np.float64)

        ytrain = train[:, 0]
        ytest = test[:, 0]

        xtrain = np.delete(train, 0, axis=1)
        xtest = np.delete(test, 0, axis=1)

        return xtrain, ytrain, xtest, ytest


    def znormalisation(self,x):

        stds = np.std(x, axis=1, keepdims=True)
        if len(stds[stds == 0.0]) > 0:
            stds[stds == 0.0] = 1.0
            return (x - x.mean(axis=1, keepdims=True)) / stds
        return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True))


    def encode_labels(self,y):

        labenc = LabelEncoder()

        return labenc.fit_transform(y)


    def preprocess(self, dataset_name):
        try:
            xtrain, ytrain, xtest, ytest = self.load_data(dataset_name)

            xtrain = self.znormalisation(xtrain)
            xtrain = np.expand_dims(xtrain, axis=2)
            #print(xtrain.shape)

            xtest = self.znormalisation(xtest)
            xtest = np.expand_dims(xtest, axis=2)
            #print(xtest.shape)

            ytrain = self.encode_labels(ytrain)
            ytest = self.encode_labels(ytest)

            return xtrain,ytrain,xtest,ytest
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None, None, None