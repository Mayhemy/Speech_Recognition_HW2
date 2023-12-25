import os
import librosa
import numpy as np
import scipy
from scipy.spatial.distance import euclidean as cosine_dist, euclidean, cosine
from concurrent.futures import ProcessPoolExecutor

from fastdtw import fastdtw

from tqdm import tqdm
import time

from scipy.signal import lfilter, hamming
from scipy.linalg import toeplitz, solve_toeplitz


N_MFCC = 13  # Number of MFCC features
N_MELS = 60  # Number of Mel bands
LPC_ORDER = 12  # Order of LPC



def dtw_from_class(dist_mat):
    N, M = dist_mat.shape
    # Initialize the cost matrix with zeros and then set the first row and column to infinity
    cost_mat = np.zeros((N + 1, M + 1))
    cost_mat[0, :] = np.inf
    cost_mat[:, 0] = np.inf
    cost_mat[0, 0] = 0

    # Initialize the traceback matrix
    traceback_mat = np.zeros((N, M))

    # Fill the cost and traceback matrices
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            penalty = [
                cost_mat[i - 1, j - 1],  # match
                cost_mat[i - 1, j],      # insertion
                cost_mat[i, j - 1]]      # deletion
            i_penalty = np.argmin(penalty)
            cost_mat[i, j] = dist_mat[i - 1, j - 1] + penalty[i_penalty]
            traceback_mat[i - 1, j - 1] = i_penalty

    # Traceback from bottom right to top left
    i, j = N - 1, M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:      # Match
            i, j = i - 1, j - 1
        elif tb_type == 1:    # Insertion
            i = i - 1
        else:                 # Deletion
            j = j - 1
        path.append((i, j))

    cost_mat = cost_mat[1:, 1:]
    alignment_cost = cost_mat[N - 1, M - 1]
    normalized_cost = alignment_cost / (N + M)

    return normalized_cost, path[::-1]


def load_audio_files(directory):
    audio_files = {}
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            digit = filename.split('-')[0]
            student_id = 'Moi'
            if len(filename.split('-')) != 1:
                student_id = filename.split('-')[1] + " " + filename.split('-')[2]
            if digit not in audio_files:
                audio_files[digit] = []
            audio, sr = librosa.load(os.path.join(directory, filename), sr=None)
            audio_files[digit].append((audio, sr, student_id))
    return audio_files


def extract_features(audio, sr, feature_type):
    if feature_type == 'mel':
        mel_spec = librosa.feature.melspectrogram(y=audio/1.0, sr=sr, n_mels=N_MELS, n_fft=256, hop_length=256)
        return np.log(mel_spec + 1e-6)
    elif feature_type == 'mfcc':
        return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=256, hop_length=256)
    elif feature_type == 'lpc':
        a = librosa.lpc(audio, order=LPC_ORDER)
        b = np.hstack([[0], -1 * a[1:]])
        y_filtered = scipy.signal.lfilter(b, [1], audio)
        mel_y_filtered = librosa.feature.melspectrogram(y=y_filtered / 1.0, sr=sr, n_mels=N_MELS, n_fft=256, hop_length=256)
        return np.log(mel_y_filtered + 1e-6)  # https://librosa.org/doc/0.8.1/generated/librosa.lpc.html ovde objasni kako se koristi ne znam zasto ovako ali eto


#  za Mel prvo nije radilo sa jako malim n_fft < 100 i malim hop_length < 20, kada sam povecao nfft drasticno na 1024 morao sam i da smanjim hop length,
# jer bi se program izvrsavao jako dugo, i pri prvim stavljenim parametrima za hop length sve radi kako treba, sto se tice Mel bendova 40 je sasvim dovoljno da se
# napravi adekvatan prediction ali je ostavljeno na 128 jer razlika u runtime-u nije drasticna

# run za 128 Mel Bandova su se prvi i drugi u najgorem slucaju razlikovali za 0.025 dok je to kod 40 mel bendova bilo 0.013




def dtw_distance(feature1, feature2):

    dist_matrix = scipy.spatial.distance.cdist(feature1.T, feature2.T, "cosine")
    distance, __ = dtw_from_class(dist_matrix)
    # distance, _ = fastdtw(feature1.T, feature2.T, dist=cosine)  # cosine jer trazimo angle izmedju vektora a ne distancu ( ako imamo dva signala jedan jaci jedan slabiji cosine distance nam je mnogo bolji)
    return distance

sum_distances_by_student1 = {}
def recognize_digit(input_audio, input_sr, reference_audios, feature_types):
    array_of_results = []

    for feature_type in feature_types:
        feature_results = {}
        for digit, audio_sr_pairs in reference_audios.items():
            print("WORKING")
            distances = []
            for audio, sr, student_id in audio_sr_pairs:
                ref_feature = extract_features(audio, sr, feature_type)
                distance = dtw_distance(extract_features(input_audio, input_sr, feature_type), ref_feature)
                distances.append(distance)
                if student_id not in sum_distances_by_student1:  #added this for comparing the most similar voice (terrible code but works)
                    sum_distances_by_student1[student_id] = 0
                sum_distances_by_student1[student_id] += distance
            feature_results[(digit, feature_type)] = sum(distances)
        array_of_results.append(feature_results)

    return array_of_results, sum_distances_by_student1


def main():
    reference_audios = load_audio_files('data')
    myInput_audios = load_audio_files('myInput')
    feature_types = ['mel', 'mfcc', 'lpc']
    output_file_path = 'output.txt'

    with open(output_file_path, 'w') as file:

        for iterator, (digit, specific_number_recordings) in enumerate(myInput_audios.items()):

            input_audio, input_sr, my_id = specific_number_recordings[0]
            recognition_results, sum_distances_by_student = recognize_digit(input_audio, input_sr, reference_audios, feature_types)

            print("Doing processing for number ------------------------------------------------- ", digit, file=file)

            for i in range(len(feature_types)):
                sorted_results = sorted(recognition_results[i].items(), key=lambda x: x[1])
                print("For feature type ", feature_types[i], file=file)
                for result in sorted_results:
                    print(f"Digit: {result[0][0]}, Feature: {result[0][1]}, Distance: {result[1]}", file=file)  # here we print for each feature and digit combination -  distance

                if sorted_results[0][1] > 0.2:  # just a threshold if its above this value we wont accept this solution
                    print("No valid digit recognized.", file=file)
                    continue
                recognized_digit = sorted_results[0][0][0]
                if recognized_digit in map(str, range(10)):
                    print(f"Recognized digit: {recognized_digit}", file=file)
                else:
                    print("No valid digit recognized.", file=file)

        # printing the id of a student with the most similar speech to mine

        min_value = min(sum_distances_by_student1.values())
        min_key = min(sum_distances_by_student1, key=sum_distances_by_student1.get)

        print("The student with the most similar speech to mine is : ", min_key, file=file)

        # testing for audio without a digit

        input_audio, input_sr = librosa.load('humanizam.wav')
        recognition_results, sum_distances_by_student = recognize_digit(input_audio, input_sr, reference_audios,
                                                                        feature_types)

        print("Doing processing for number ", "none", file=file)

        # Process and display results
        for i in range(len(recognition_results)):
            sorted_results = sorted(recognition_results[i].items(), key=lambda x: x[1])
            print("For feature type ", feature_types[i])
            for result in sorted_results:
                print(f"Digit: {result[0][0]}, Feature: {result[0][1]}, Distance: {result[1]}", file=file)

            if sorted_results[0][1] > 0.3:
                print("No valid digit recognized.", file=file)
                continue
            recognized_digit = sorted_results[0][0][0]
            if recognized_digit in map(str, range(10)):
                print(f"Recognized digit: {recognized_digit}", file=file)
            else:
                print("No valid digit recognized.", file=file)



if __name__ == "__main__":
    main()
