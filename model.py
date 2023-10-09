import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa

# Load data
inputs = np.load(r'all_magnitudes_array.npy')
outputs = np.load(r'concatenated_onehot_data.npy')

print("Shape of inputs:", inputs.shape)
print("Shape of outputs:", outputs.shape)

# Reshape the data to fit the convolutional layer input
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
X_train_reshaped = X_train.reshape(-1, 401, 1)
X_test_reshaped = X_test.reshape(-1, 401, 1)

# Split y_train and y_test into their respective categories
y_train_base = y_train[:, :12]
y_train_octave = y_train[:, 12:20]
y_train_chord = y_train[:, 20:]

y_test_base = y_test[:, :12]
y_test_octave = y_test[:, 12:20]
y_test_chord = y_test[:, 20:]

# Model definition
input_layer = keras.layers.Input(shape=(401, 1))
x = keras.layers.Conv1D(64, kernel_size=5, activation='elu')(input_layer)
x = keras.layers.Conv1D(128, kernel_size=3, activation='elu')(x)
x = keras.layers.Conv1D(64, kernel_size=5, activation='elu')(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(500, activation='elu')(x)
x = keras.layers.Dropout(rate=0.7)(x)
x = keras.layers.Dense(500, activation='elu')(x)
x = keras.layers.Dropout(rate=0.7)(x)
x = keras.layers.Dense(200, activation='elu')(x)

base_note_output = keras.layers.Dense(12, activation='softmax', name='base_note')(x)
octave_output = keras.layers.Dense(8, activation='softmax', name='octave')(x)
chord_type_output = keras.layers.Dense(24, activation='softmax', name='chord_type')(x)

model = keras.models.Model(inputs=input_layer, outputs=[base_note_output, octave_output, chord_type_output])

# Compilation
model.compile(optimizer='adam',
              loss={'base_note': 'binary_crossentropy',
                    'octave': 'binary_crossentropy',
                    'chord_type': 'binary_crossentropy'},
              metrics={'base_note': 'accuracy',
                       'octave': 'accuracy',
                       'chord_type': 'accuracy'},
              loss_weights = {'base_note': 2.0, 'octave': 1.0, 'chord_type': 3.0})

#load weights
model.load_weights('3c3d_elu_model.h5')

# Shuffling training data
indices = np.arange(X_train_reshaped.shape[0])
np.random.shuffle(indices)
X_train_reshaped = X_train_reshaped[indices]
y_train_base = y_train_base[indices]
y_train_octave = y_train_octave[indices]
y_train_chord = y_train_chord[indices]

# Training
history = model.fit(X_train_reshaped, [y_train_base, y_train_octave, y_train_chord],
                    epochs=50, validation_split=0.1, batch_size=500)

# Evaluation
test_loss = model.evaluate(X_test_reshaped, [y_test_base, y_test_octave, y_test_chord], verbose=2)
print("\nBase Note Test Accuracy:", test_loss[4])
print("\nOctave Test Accuracy:", test_loss[5])
print("\nChord Type Test Accuracy:", test_loss[6])

# Saving model weights
model.save_weights('3c3d_elu_model.h5')
