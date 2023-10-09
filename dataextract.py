import os
import numpy as np
import soundfile as sf
from scipy.signal import windows
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import gpt42


main_directory = r'C:\Users\faumb\OneDrive\Desktop\chords'

file_names = []
all_magnitudes = []

stride_seconds = 0.05
window_duration = 0.05

for dirpath, dirnames, filenames in os.walk(main_directory):
    for filename in filenames:
        if filename.endswith('.wav'):
            wav_path = os.path.join(dirpath, filename)

            # Load the WAV file using soundfile
            data, sample_rate = sf.read(wav_path)

            # Ensure the data is float type
            data = np.asarray(data, dtype=np.float64)

            # Convert stereo to mono, if necessary
            if len(data.shape) == 2:
                data = np.mean(data, axis=1)

            start_time = 0.01
            samples = 0
            while (start_time + window_duration) <= (len(data) / sample_rate) and samples <= 7:
                samples+=1
                start_index = int(start_time * sample_rate)
                end_index = start_index + int(window_duration * sample_rate)

                segment = data[start_index:end_index]

                window = windows.hann(len(segment))
                windowed_segment = segment * window

                magnitude = np.abs(np.fft.rfft(windowed_segment))
                all_magnitudes.append(magnitude)

                file_names.append(filename)

                start_time += stride_seconds


all_magnitudes_array = np.array(all_magnitudes)

mean_magnitudes = np.mean(all_magnitudes_array, axis=0)
std_magnitudes = np.std(all_magnitudes_array, axis=0)

all_magnitudes_array = (all_magnitudes_array - mean_magnitudes) / std_magnitudes

print(all_magnitudes_array)
print(len(all_magnitudes_array))

gpt42.plot(2, all_magnitudes_array, sample_rate)


####
def parse_filename(filename):
    base_name = filename[:-4]
    parts = base_name.split('-')
    return parts[0], parts[1], '-'.join(parts[2:-1]), parts[-1]


parsed_data = [parse_filename(fname) for fname in file_names]

# Splitting the parsed data
notes, octaves, chord_types, variants = zip(*parsed_data)


# One-hot encode function
def one_hot_encode(data):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = label_encoder.fit_transform(data).reshape(-1, 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    return onehot_encoded

notes_onehot = one_hot_encode(notes)
octaves_onehot = one_hot_encode(octaves)
chord_types_onehot = one_hot_encode(chord_types)
#variants_onehot = one_hot_encode(variants) dont include because variations are arbitrary
concatenated_onehot_data = [np.concatenate((n, o, c)) for n, o, c in zip(notes_onehot, octaves_onehot, chord_types_onehot)]
concatenated_onehot_data_array = np.array(concatenated_onehot_data)

print(len(notes_onehot[0]))
print(len(octaves_onehot[0]))
print(len(chord_types_onehot[0]))

np.save('concatenated_onehot_data.npy', concatenated_onehot_data_array)
np.save('all_magnitudes_array.npy', all_magnitudes_array)

