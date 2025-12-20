# last update: 2025.12.20 by fjl
import os
import random
import pandas as pd
import torch
from torchvision import transforms
import numpy as np
from image_transformations_2 import test_flip_left_right, mr_gaussian, test_rotate, mr_colored, mr_brightness
from sklearn.svm import SVC
from tqdm import tqdm
import pickle
from joblib import Parallel, delayed
from GoogLeNet_model import model, device
import time


start_time = time.time()

# Parameters set up
data_dir = '~'  # Your image path, this file only has images, no subfolder
output_excel = "~/xx.xlsx"
label_excel_path = '~/xx.xlsx'
computational_process_times = 5  # K for computational processes and K-2 for optimization
total_select_MPs = 1000  # MPs selection number
fail_pair_count = 0  # failed MPs count
total_DNN_call = 0  # DNN calling times

# Save failed MPs details
test_results = []
is_last = False  # The last computational process does not update surrogate model
n_experiments = 5  # The number of repetitions of the experiment, can be set by yourself.

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # GoogLeNet,ResNet50 use (224,224)
    # transforms.Resize((299, 299)),  # Inception V3 uses (299,299)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

def load_labels_from_excel(excel_path, image_paths, M):
    # load excel
    df = pd.read_excel(excel_path)
    image_names = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    # Establish a mapping from file names to indexes
    name_to_index = {name: i for i, name in enumerate(image_names)}
    N = len(image_names)
    Label_0 = [-1] * N
    Label = [[-1] * N for _ in range(M)]
    # for each row
    for _, row in df.iterrows():
        name = str(row['filename'])
        if name in name_to_index:
            idx = name_to_index[name]
            Label_0[idx] = int(row['labelSource'])
            for j in range(M):
                Label[j][idx] = int(row[f'labelFollowUp_{j + 1}'])
    return Label_0, Label

# Image paths
image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)
               if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

# Define MRs
MRs = [test_flip_left_right, mr_gaussian, test_rotate, mr_colored, mr_brightness]

# Call DNN for test
def get_label(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
    return pred.item()

# Read saved data (original images and latent space features)
# "Saved_Data_10" for CIFAR-10
# "Svaed_Data_Pets" for Oxford-IIIT Pet
output_dir = os.path.join(os.getcwd(), '/Volumes/My Passport/Save_mnist_images')

def load_data_from_file(output_dir):
    file_paths = {
        "P_0": os.path.join(output_dir, "P_0.pkl"),
        "P": os.path.join(output_dir, "P.pkl"),
        "P_0_features": os.path.join(output_dir, "P_0_features.pkl"),
        "P_features": os.path.join(output_dir, "P_features.pkl"),
    }
    data = {}
    for key, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Can not find {path}，please generate and save data first!")
        with open(path, "rb") as f:
            data[key] = pickle.load(f)
    return data["P_0"], data["P"], data["P_0_features"], data["P_features"]

# Load data
P_0, P, P_0_features, P_features = load_data_from_file(output_dir)

# Initialize Label_0 and Label lists
Label_0 = [-1] * len(P_0)
Label = [[-1] * len(P[i]) for i in range(len(P))]

Source_label, Follow_label = load_labels_from_excel(label_excel_path, image_paths, len(MRs))

# Randomly choose and initialize surrogate model
# Select source test cases and match a MR
num_source_images = total_select_MPs // computational_process_times
selected_sources = random.sample(range(len(P_0)), num_source_images)
each_MPs = total_select_MPs // computational_process_times  # MPs selection number for this progress

# Initialize temp lists
source_features = []
follow_up_features = []
source_labels = []
follow_up_labels = []
each_DNN = 0

print("Getting first random data...")
for i in selected_sources:
    source_image = P_0[i]  # get source original images
    source_label = Source_label[i]  # call DNN for source labels
    Label_0[i] = source_label  # save source DNN labels

    # Randomly choose a MP and call DNN for its label
    mr_index = random.choice(range(len(MRs)))
    follow_up_image = P[mr_index][i]
    follow_up_label = Follow_label[mr_index][i]  # get follow_up cases DNN labels
    Label[mr_index][i] = follow_up_label  # save follow_up DNN labels

    # If follow_up label is different from source label, failed MPs + 1
    if source_label != follow_up_label:
        fail_pair_count += 1
        test_results.append([os.path.basename(image_paths[i]), mr_index, Label_0[i], Label[mr_index][i]])

    # Get the latent space feature of source and follow_up cases
    source_features.append(P_0_features[i])
    follow_up_features.append(P_features[mr_index][i])
    source_labels.append(source_label)
    follow_up_labels.append(follow_up_label)

    total_DNN_call += 2  # DNN calling times is 2
    each_DNN += 2

print("Training the initial model of SVM...")
# Train a surrogate model
X_train = np.array(source_features + follow_up_features)
X_train = X_train.reshape(X_train.shape[0], -1)
y_train = np.array(source_labels + follow_up_labels)

clf = SVC(kernel='rbf', gamma='scale', coef0=1)  # use sigmoid
clf.fit(X_train, y_train)
print("The initial model has been trained successfully!")

# Output of the first random selection
print(f"The number of failed MPS found in the first random selection: {fail_pair_count}")
print(f"DNN calling times in the first random selection: {each_DNN}")

# parallel computing SVM model prediction
def predict_svm(i, features, model):
    pred = model.predict([features.flatten()])[0]
    dist = np.min(abs(model.decision_function([features.flatten()])[0]))
    return i, pred, dist

def UpdateSVM(P_0, P, P_0_features, P_features, Label_0, Label, SVM, MRs, each_MPs, total_DNN_call, is_last):
    iteration_MPs = each_MPs
    each_DNN_call = 0
    error_num = 0
    test_results_temp = []

    SVM_Label_0 = np.full_like(Label_0, -1)
    SVM_Label = np.full_like(Label, -1)
    SVM_Dis_0 = np.full_like(Label_0, np.inf, dtype=float)
    SVM_Dis = np.full_like(Label, np.inf, dtype=float)

    print("Processing Source Images...")
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(predict_svm)(i, P_0_features[i], SVM) for i in range(len(Label_0)) if Label_0[i] == -1)
    for i, pred, dist in results:
        SVM_Label_0[i] = pred
        SVM_Dis_0[i] = dist

    print("Processing Follow-up Images...")
    for j in tqdm(range(len(MRs)), desc="Processing Follow-up Images (MRs)"):
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(predict_svm)(i, P_features[j][i], SVM) for i in range(len(Label[j])) if Label[j][i] == -1)
        for i, pred, dist in results:
            SVM_Label[j][i] = pred
            SVM_Dis[j][i] = dist

    M = len(MRs)
    N = len(P_0)
    Ave_Dis = np.full((M, N), np.inf)

    for i in range(N):
        for j in range(M):
            if SVM_Label_0[i] == -1 and SVM_Label[j][i] == -1:
                continue
            elif SVM_Label_0[i] == -1 and SVM_Label[j][i] != -1:
                if SVM_Label[j][i] != Label_0[i]:
                    Ave_Dis[j, i] = SVM_Dis[j][i]
            elif SVM_Label_0[i] != -1 and SVM_Label[j][i] != -1:
                if SVM_Label_0[i] != SVM_Label[j][i]:
                    Ave_Dis[j, i] = (SVM_Dis_0[i] + SVM_Dis[j][i]) / 2

    if is_last == False:
        indices = np.argsort(Ave_Dis, axis=None)
        sorted_positions = np.unravel_index(indices, Ave_Dis.shape)
        Sorted_Ave_Dis = list(zip(sorted_positions[0], sorted_positions[1]))
    else:
        finite_mask = np.isfinite(Ave_Dis)
        indices = np.argsort(-Ave_Dis[finite_mask])
        valid_positions = np.column_stack(np.where(finite_mask))
        sorted_positions = valid_positions[indices]
        Sorted_Ave_Dis = [tuple(pos) for pos in sorted_positions]

    source_features, source_labels = [], []
    follow_up_features, follow_up_labels = [], []

    index = 0
    while iteration_MPs > 0 and index < len(Sorted_Ave_Dis):
        i, j = Sorted_Ave_Dis[index]  # i: MR index, j: source index

        if Label_0[j] == -1:
            Label_0[j] = Source_label[j]
            source_features.append(P_0_features[j])
            source_labels.append(Label_0[j])
            total_DNN_call += 1
            each_DNN_call += 1
            if iteration_MPs == 0:
                break

        if Label[i][j] == -1:
            Label[i][j] = Follow_label[i][j]
            follow_up_features.append(P_features[i][j])
            follow_up_labels.append(Label[i][j])
            iteration_MPs -= 1
            total_DNN_call += 1
            each_DNN_call += 1

        if Label_0[j] != Label[i][j]:
            error_num += 1
            test_results_temp.append([os.path.basename(image_paths[j]), i, Label_0[j], Label[i][j]])

        index += 1

    return source_features, source_labels, follow_up_features, follow_up_labels, error_num, test_results_temp, total_DNN_call, each_DNN_call

print("Start iteratively updating SVM...")
for iteration in range(computational_process_times - 2):
    print(f"The {iteration + 1} time for SVM optimization...")
    (new_source_features, new_source_labels, new_follow_up_features, new_follow_up_labels,
     error_num, test_results_temp, total_DNN_call, each_DNN) = UpdateSVM(
        P_0, P, P_0_features, P_features, Label_0, Label, clf, MRs, each_MPs, total_DNN_call, is_last)

    fail_pair_count += error_num
    source_features.extend(new_source_features)
    follow_up_features.extend(new_follow_up_features)
    source_labels.extend(new_source_labels)
    follow_up_labels.extend(new_follow_up_labels)
    test_results.extend(test_results_temp)

    X_train = np.array(source_features + follow_up_features)
    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = np.array(source_labels + follow_up_labels)
    clf = SVC(kernel='rbf', gamma='scale', coef0=1)
    clf.fit(X_train, y_train)

    print(f"The {iteration + 1} time for SVM optimization finished")
    print(f"The {iteration + 1} time finds {error_num} failed MPs")
    print(f"The {iteration + 1} time calls DNN {each_DNN} times")

print("The last process for MPs selection")
is_last = True
(new_source_features, new_source_labels, new_follow_up_features, new_follow_up_labels,
 error_num, test_results_temp, total_DNN_call, each_DNN) = UpdateSVM(
    P_0, P, P_0_features, P_features, Label_0, Label, clf, MRs, each_MPs, total_DNN_call, is_last)

fail_pair_count += error_num
test_results.extend(test_results_temp)

print(f"The final selection finished")
print(f"The final selection finds {error_num} failed MPs")
print(f"The final selection calls DNN {each_DNN} times")

# Save to Excel
df = pd.DataFrame(test_results, columns=["Source Image", "MR Index", "Source Label", "Follow-up Label"])
df.to_excel(output_excel, index=False)
print("Saved to UpdateSVM_metamorphic_test_results.xlsx")

# Output the final result
print(f"Total failed MPs: {fail_pair_count}/{total_select_MPs}")
print(f"Total DNN calls: {total_DNN_call}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total time: {elapsed_time:.4f} seconds")
