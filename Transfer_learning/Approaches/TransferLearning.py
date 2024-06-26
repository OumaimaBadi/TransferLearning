import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, BatchNormalization, Input, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from utils.utils import *

class SaveBestModel(Callback):
    def __init__(self, save_path):
        super(SaveBestModel, self).__init__()
        self.best_weights = None
        self.save_path = save_path
        self.best_loss = np.Inf
        self.previous_file = None

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("loss")
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_weights = self.model.get_weights()
            # Delete previous file if it exists
            if self.previous_file and os.path.exists(self.previous_file):
                os.remove(self.previous_file)
            self.previous_file = self.save_path
            self.model.save(self.save_path)  # Save in HDF5 format

class TransferLearning:
    def __init__(self, models_directory, pretrained_models_directory, features_directory):
        self.models_directory = models_directory
        self.pretrained_models_directory = pretrained_models_directory
        self.features_directory = features_directory

    def reset_batch_norm_layers(self, model):
        """Utility function to reset batch normalization layers."""
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                for attr in ['moving_mean', 'moving_variance']:
                    if hasattr(layer, attr):
                        var = getattr(layer, attr)
                        var.assign(tf.zeros_like(var))
                if hasattr(layer, 'gamma'):
                    layer.gamma.assign(tf.keras.initializers.glorot_uniform()(layer.gamma.shape))
                if hasattr(layer, 'beta'):
                    layer.beta.assign(tf.zeros_like(layer.beta))

    def modify_model_for_transfer_learning(self, source_model, input_shape):
        tf.keras.backend.clear_session()
        input_layer = Input(shape=(input_shape,))
        reshape_layer = Reshape(target_shape=(input_shape, 1))(input_layer)

        # Define the function to copy the configuration and weights of each layer
        def copy_layer(source_layer, new_input):
            config = source_layer.get_config()
            new_layer = source_layer.__class__.from_config(config)
            new_layer.build(new_input.shape)
            new_layer.set_weights(source_layer.get_weights())
            return new_layer(new_input)

        # Apply the layers up to the first Concatenate layer
        hybrid_increase_outputs = []
        hybrid_decrease_outputs = []
        hybrid_peeks_outputs = []

        for i in range(6):
            layer_name = f'hybird-increasse-{i}-{2**(i+1)}'
            conv_layer = source_model.get_layer(layer_name)
            hybrid_increase_outputs.append(copy_layer(conv_layer, reshape_layer))
        
        for i in range(6, 12):
            layer_name = f'hybird-decrease-{i}-{2**(i-5)}'
            conv_layer = source_model.get_layer(layer_name)
            hybrid_decrease_outputs.append(copy_layer(conv_layer, reshape_layer))
        
        for i in range(12, 17):
            layer_name = f'hybird-peeks-{i}-{2**(i-11+1)}'
            conv_layer = source_model.get_layer(layer_name)
            hybrid_peeks_outputs.append(copy_layer(conv_layer, reshape_layer))

        concatenated_hybrid = Concatenate()(
            hybrid_increase_outputs + hybrid_decrease_outputs + hybrid_peeks_outputs
        )

        # Apply the additional Conv1D layers
        conv1d_layers = []
        conv1d_layers.append(copy_layer(source_model.get_layer('conv1d'), reshape_layer))
        conv1d_layers.append(copy_layer(source_model.get_layer('conv1d_1'), reshape_layer))
        conv1d_layers.append(copy_layer(source_model.get_layer('conv1d_2'), reshape_layer))

        activation_layer = source_model.get_layer('activation')(concatenated_hybrid)

        concatenated_all = Concatenate()(
            conv1d_layers + [activation_layer]
        )

        # Apply the remaining layers after the second Concatenate layer
        x = concatenated_all
        for layer in source_model.layers[source_model.layers.index(source_model.get_layer('concatenate_1')) + 1:]:
            x = copy_layer(layer, x)

        modified_model = Model(inputs=input_layer, outputs=x)
        # Reset batch normalization layers
        self.reset_batch_norm_layers(modified_model)

        return modified_model

    def training(self, source_dataset_name, target_dataset_name, use_catch22_features=True):
        try:
            target_data = load_and_prepare_data(target_dataset_name, self.features_directory)
            if target_data is None:
                return None

            X_train, y_train, _, _, train_features, _ = target_data

            y_train_categorical = tf.keras.utils.to_categorical(y_train)
            input_shape = X_train.shape[1]
            print(input_shape)
            n_classes = len(np.unique(y_train))

            source_models = load_latent_models(self.pretrained_models_directory, source_dataset_name)

            for i, source_model in enumerate(source_models):
                # Ensure TensorFlow graph is cleared and reset
                tf.keras.backend.clear_session()

                #source_model.summary()
                modified_model = self.modify_model_for_transfer_learning(source_model, input_shape)
                #modified_model.save('modified.hdf5')
                #modified_model.summary()

                if use_catch22_features:
                    custom_input = Input(shape=(train_features.shape[1],), name='Catch22_input')
                    concatenated_output = Concatenate(name='concatenate_LS_Catch22')([modified_model.output, custom_input])
                    concatenated_output = BatchNormalization(name='batch_normalization_LS_Catch22')(concatenated_output)
                else:
                    concatenated_output = modified_model.output

                classifier_output = Dense(n_classes, activation='softmax')(concatenated_output)
                final_inputs = [modified_model.input, custom_input] if use_catch22_features else modified_model.input
                final_model = Model(inputs=final_inputs, outputs=classifier_output)

                target_source_dir = os.path.join(self.models_directory, target_dataset_name, source_dataset_name)
                os.makedirs(target_source_dir, exist_ok=True)

                file_path = os.path.join(target_source_dir, f'{target_dataset_name}_best_model_{i}.h5')
                plot_path = os.path.join(target_source_dir, f'{target_dataset_name}_hist_{i}.pdf')

                save_best_model = SaveBestModel(save_path=file_path)

                callbacks = [
                    tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=50, min_lr=0.0001),
                    save_best_model
                ]

                final_model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
                history = final_model.fit([X_train, train_features], y_train_categorical, epochs=1500, batch_size=64, callbacks=callbacks)

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(history.history['loss'], label='Training Loss')
                plt.subplot(1, 2, 2)
                plt.plot(history.history['accuracy'], label='Training Accuracy')
                plt.savefig(plot_path)
                plt.close()

        except Exception as e:
            print(f"Error during training for target dataset {target_dataset_name}: {e}")

    def evaluate_models(self, dataset_name):
        return evaluate_models(dataset_name, self.models_directory, self.features_directory)
    
    def rotation_forest_ensembling(self, dataset_name):
        return evaluate_rotation_forest_ensembling(dataset_name, self.models_directory, self.features_directory)

# Example usage
# if __name__ == "__main__":
#     a = "models_3"
#     b = "/home/oumaima/Transfer_learning/LITE"
#     c = "/home/oumaima/Transfer_learning/catch22_features"

#     transfer_learning = TransferLearning(models_directory=a, pretrained_models_directory=b, features_directory=c)
#     transfer_learning.training(source_dataset_name='ArrowHead', target_dataset_name='ACSF1')
