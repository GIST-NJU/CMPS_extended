import scipy.io
import os
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
from scipy.spatial.distance import euclidean
import mr_5
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageOps
import my_dbscan
import math
import random
import pickle
import time



def load_and_preprocess_images(img_paths):
    img_array_list = []
    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        img_array_list.append(img_array)
    return np.vstack(img_array_list)


def get_image_paths_from_directory(directory):
    valid_extensions = ('.jpg', '.jpeg', '.png')
    img_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.lower().endswith(valid_extensions)]
    return img_paths

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def remove_duplicates(pairs):
    unique_pairs = set()
    for pair in pairs:
        unique_pairs.add(tuple(pair))
    return list(unique_pairs)

def data_process(path):
    ori_mdata = [None, None,None]
    mr_mdata = []

    for mr in mr_list:
        file = scipy.io.loadmat(str(path) + mr + '_classification_results.mat')
        cell_data_list = []
        cell_dict = {}
        for cell in cell_list:
            variable_name = cell
            if variable_name in file:
                variable_data = file[variable_name][()]
                if variable_name == 'confidenceSourceArray':
                    confidenceSourceArray = []
                    for item in variable_data:
                        confidenceSourceArray.append(list(item[0][0]))
                    ori_mdata[2] = confidenceSourceArray
                elif variable_name == 'confidenceFollowUpArray':
                    filenames = []
                    for item in variable_data:
                        filenames.append(list(item[0][0]))
                    cell_data_list.append(filenames)
                elif variable_name == 'labelSourceArray':
                    labelSourceArray = [item[0][0] if (item is not None and item[0].size > 0) else ' ' for item in variable_data]
                    ori_mdata[1] = labelSourceArray
                elif variable_name == 'labelFollowUpArray':
                    filenames = [item[0][0] if (item is not None and item[0].size > 0) else ' ' for item in variable_data]
                    cell_data_list.append(filenames)
                else:
                    filenames = [item[0][0] for item in variable_data]
                    ori_mdata[0] = filenames
                    cell_data_list.append(filenames)
            else:
                print(f"Variable '{variable_name}' not found in the .mat file.")
        for index1 in range(len(cell_data_list[0])):
            image_path = base_path + cell_data_list[0][index1]
            cell_dict[image_path] = [cell_data_list[1][index1]]
            for index2 in range(2, len(cell_data_list)):
                cell_dict[image_path].append(cell_data_list[index2][index1])
        mr_mdata.append(cell_dict)
    return ori_mdata,mr_mdata

def deepgini(outputs):
    sum_un=0
    for i in outputs:
       if i > 0:
           sum_un += pow(i, 2)
    uncertainty=1-sum_un
    return uncertainty

def margin(outputs):
    """Prediction margin: difference between top-1 and top-2 probabilities."""
    sorted_p = sorted(outputs, reverse=True)
    if len(sorted_p) > 1:
        return sorted_p[0] - sorted_p[1]
    return 1.0

def maxp(outputs):
    """Prediction maxp: top-1 probability."""
    sorted_p = sorted(outputs, reverse=True)
    return 1 - sorted_p[0]


# CMPS++ full version: Harmonic Mean (DeepGini, MaxP) + MA2-A
def CMPS_extend(ori_mdata, mr_mdata, budget):
    # clustering
    predictions = ori_mdata[2]
    if dataset == 'imagenet':
        n = 1000
    elif dataset == 'fruit360':
        n = 141
    else:
        n = 10
    predictions = np.array([np.pad(pred, (0, n - len(pred)), 'constant') for pred in predictions])
    scaler = StandardScaler()
    predictions_scaled = scaler.fit_transform(predictions)
    dbscan_labels = my_dbscan.dbscan(predictions_scaled, eps=0.15, min_samples=1)

    cluster_counts = Counter(dbscan_labels)
    cluster_size = {}
    for cluster_label, count in cluster_counts.items():
        cluster_size[cluster_label] = count
    image_cluster_count = {}
    for k in dbscan_labels:
        if k not in image_cluster_count:
            image_cluster_count[k] = 0

    imgpath_pred_dict = {}
    imgpath_orilabel_dict = {}
    image_uncertainty = dict()
    image_cluster = {}

    # Harmonic Mean of DeepGini and MaxP
    for index in range(len(img_paths)):
        image_cluster[img_paths[index]] = dbscan_labels[index]
        imgpath_pred_dict[img_paths[index]] = predictions[index]
        ori_label = ori_mdata[1][index]
        imgpath_orilabel_dict[img_paths[index]] = ori_label

        gini = deepgini(predictions[index])
        mp = 1.0 - max(predictions[index])  # MaxP as uncertainty
        image_uncertainty[img_paths[index]] = 2 * gini * mp / (gini + mp + 1e-10)

    image_uncertainty_sorted = sorted(image_uncertainty.items(), key=lambda x: x[1], reverse=True)

    # Build img_path -> index mapping for loading source images
    imgpath_to_idx = {p: i for i, p in enumerate(img_paths)}

    # start selection
    processed_img = []
    fault_record = {}
    select_record = {}
    model_run_times = 0
    error_num = 0
    d = 0

    good_clus_list = []
    bad_clus_list = []
    pre_good = []
    pre_bad = []
    round = 0
    file_budget = budget

    # MA2-A global state variables
    PRIOR_HITS, PRIOR_TRIES = 1.0, 5.0
    hits_per_mr = [PRIOR_HITS] * 5
    tries_per_mr = [PRIOR_TRIES] * 5
    seen_unique = set()
    sources_with_new_pair = 0
    total_sources_done = 0
    total_pairs_found = 0
    max_k = 5
    marginal_unique_at_k = [1.0] * max_k
    marginal_tries_at_k = [2.0] * max_k

    while budget > 0:

        flag = False

        for img_index in range(len(image_uncertainty_sorted)):

            img_path = image_uncertainty_sorted[img_index][0]
            img_clus = image_cluster[img_path]
            if img_clus in image_cluster_count:
                if round == 0:
                    clus_count = image_cluster_count[img_clus]
                    if img_path not in processed_img and clus_count <= d:
                        processed_img.append((img_path))
                        image_cluster_count[img_clus] += 1
                        if image_cluster_count[img_clus] == cluster_size[img_clus]:
                            del image_cluster_count[img_clus]
                        flag = True
                        break
                else:
                    clus_count = image_cluster_count[img_clus]
                    if len(pre_good) > 0:
                        if img_path not in processed_img and clus_count <= d and img_clus in pre_good:
                            processed_img.append((img_path))
                            image_cluster_count[img_clus] += 1
                            pre_good.remove(img_clus)
                            if image_cluster_count[img_clus] == cluster_size[img_clus]:
                                del image_cluster_count[img_clus]
                            flag = True
                            break
                    else:
                        if img_path not in processed_img and clus_count <= d and img_clus in pre_bad:
                            processed_img.append((img_path))
                            image_cluster_count[img_clus] += 1
                            pre_bad.remove(img_clus)
                            if image_cluster_count[img_clus] == cluster_size[img_clus]:
                                del image_cluster_count[img_clus]
                            flag = True
                            break
        if not flag:
            print("no valid image")

        # --- Load source image for MR transformation ---
        if dataset == 'fashion':
            src_img = Image.fromarray(img_arrays[imgpath_to_idx[img_path]].astype(np.uint8))
        else:
            src_img = Image.open(img_path)

        # --- MA2-A MR selection logic ---
        ori_label = imgpath_orilabel_dict[img_path]

        # Compute Bayesian estimates
        source_yield_rate = (sources_with_new_pair + 1) / (total_sources_done + 2)
        pair_uniqueness_rate = (len(seen_unique) + 1) / (total_pairs_found + 2)

        # Conditional decay
        if pair_uniqueness_rate < source_yield_rate:
            decay = 1.0 - (1.0 - source_yield_rate) * pair_uniqueness_rate
        else:
            decay = source_yield_rate

        # Sort MRs by discovery rate
        discovery_rate = [hits_per_mr[m] / tries_per_mr[m] for m in range(5)]
        next_max = max(discovery_rate)
        order = sorted(range(5), key=lambda m: discovery_rate[m], reverse=True)
        remaining = list(order)

        k = 0
        any_violation = False
        found_new_pair_this_source = False

        while remaining and budget > 0:
            mr_idx = remaining.pop(0)

            # Apply MR transformation to generate follow-up image
            # The validity of all follow-ups has been checked by SSIM and manual analysis
            follow_up_img = mr_5.test_mrs(src_img, mr_list2[mr_idx])

            # Get follow-up label
            if dataset == 'fashion':
                mr_key = base_path + img_path
            else:
                mr_key = img_path
            f_label = mr_mdata[mr_idx][mr_key][0]
            model_run_times += 1
            budget -= 1
            k += 1
            tries_per_mr[mr_idx] += 1

            violated = (f_label != ori_label)
            new_unique_this_step = False

            if violated:
                any_violation = True
                error_num += 1
                total_pairs_found += 1
                pair = (ori_label, f_label)
                if pair not in seen_unique:
                    seen_unique.add(pair)
                    hits_per_mr[mr_idx] += 1
                    found_new_pair_this_source = True
                    new_unique_this_step = True

                if img_path not in fault_record:
                    fault_record[img_path] = [(mr_list[mr_idx], ori_label, f_label)]
                else:
                    exists = any((ori == ori_label and f == f_label) for _, ori, f in fault_record[img_path])
                    if not exists:
                        fault_record[img_path].append((mr_list[mr_idx], ori_label, f_label))

            # Record in select_record
            if img_path not in select_record:
                select_record[img_path] = [(mr_list[mr_idx], ori_label, f_label)]
            else:
                select_record[img_path].append((mr_list[mr_idx], ori_label, f_label))

            # Update marginal uniqueness for position k
            marginal_tries_at_k[k - 1] += 1
            if new_unique_this_step:
                marginal_unique_at_k[k - 1] += 1

            # Stopping rules
            if not remaining:
                break

            # 1. Marginal uniqueness stopping
            next_k = k
            if next_k < max_k:
                marginal_rate = marginal_unique_at_k[next_k] / marginal_tries_at_k[next_k]
                if marginal_rate < source_yield_rate * 0.5:
                    break

            # 2. Decay stopping
            cur_max = max(hits_per_mr[m] / tries_per_mr[m] for m in remaining)
            if cur_max < next_max * decay:
                break

        # Good/Bad cluster classification
        if k > 0:
            if any_violation and image_cluster[img_path] in image_cluster_count:
                good_clus_list.append(image_cluster[img_path])
            elif not any_violation and image_cluster[img_path] in image_cluster_count:
                bad_clus_list.append(image_cluster[img_path])

        if found_new_pair_this_source:
            sources_with_new_pair += 1
        total_sources_done += 1

        flag_c = True
        for k, v in image_cluster_count.items():
            if v <= d:
                flag_c = False
        if flag_c:
            d += 1
            round += 1
            pre_good = good_clus_list
            pre_bad = bad_clus_list
            good_clus_list = []
            bad_clus_list = []

    trc = error_num / model_run_times
    print("TRC:", trc)
    all_pairs = []
    for v in fault_record.values():
        for e in v:
            all_pairs.append(e[1:])
    unique_pairs = remove_duplicates(all_pairs)
    if dataset == 'imagenet' or dataset == 'fruit360':
        faults = file_budget
    else:
        faults = 90
    fdr = len(unique_pairs) / faults
    print("FDR:", fdr)

    return trc, fdr


# CMPS++(-I1): DeepGini only + MA2-A (ablation: remove composite)
def CMPS_v1(ori_mdata, mr_mdata, budget):
    # clustering
    predictions = ori_mdata[2]
    if dataset == 'imagenet':
        n = 1000
    elif dataset == 'fruit360':
        n = 141
    else:
        n = 10
    predictions = np.array([np.pad(pred, (0, n - len(pred)), 'constant') for pred in predictions])
    scaler = StandardScaler()
    predictions_scaled = scaler.fit_transform(predictions)
    dbscan_labels = my_dbscan.dbscan(predictions_scaled, eps=0.15, min_samples=1)

    cluster_counts = Counter(dbscan_labels)
    cluster_size = {}
    for cluster_label, count in cluster_counts.items():
        cluster_size[cluster_label] = count
    image_cluster_count = {}
    for k in dbscan_labels:
        if k not in image_cluster_count:
            image_cluster_count[k] = 0

    imgpath_pred_dict = {}
    imgpath_orilabel_dict = {}
    image_uncertainty = dict()
    image_cluster = {}

    # Ablation: DeepGini only (no margin)
    for index in range(len(img_paths)):
        image_cluster[img_paths[index]] = dbscan_labels[index]
        imgpath_pred_dict[img_paths[index]] = predictions[index]
        ori_label = ori_mdata[1][index]
        imgpath_orilabel_dict[img_paths[index]] = ori_label

        image_uncertainty[img_paths[index]] = deepgini(predictions[index])

    image_uncertainty_sorted = sorted(image_uncertainty.items(), key=lambda x: x[1], reverse=True)

    imgpath_to_idx = {p: i for i, p in enumerate(img_paths)}

    processed_img = []
    fault_record = {}
    select_record = {}
    model_run_times = 0
    error_num = 0
    d = 0

    good_clus_list = []
    bad_clus_list = []
    pre_good = []
    pre_bad = []
    round = 0
    file_budget = budget

    # MA2-A global state variables
    PRIOR_HITS, PRIOR_TRIES = 1.0, 5.0
    hits_per_mr = [PRIOR_HITS] * 5
    tries_per_mr = [PRIOR_TRIES] * 5
    seen_unique = set()
    sources_with_new_pair = 0
    total_sources_done = 0
    total_pairs_found = 0
    max_k = 5
    marginal_unique_at_k = [1.0] * max_k
    marginal_tries_at_k = [2.0] * max_k

    while budget > 0:

        flag = False

        for img_index in range(len(image_uncertainty_sorted)):

            img_path = image_uncertainty_sorted[img_index][0]
            img_clus = image_cluster[img_path]
            if img_clus in image_cluster_count:
                if round == 0:
                    clus_count = image_cluster_count[img_clus]
                    if img_path not in processed_img and clus_count <= d:
                        processed_img.append((img_path))
                        image_cluster_count[img_clus] += 1
                        if image_cluster_count[img_clus] == cluster_size[img_clus]:
                            del image_cluster_count[img_clus]
                        flag = True
                        break
                else:
                    clus_count = image_cluster_count[img_clus]
                    if len(pre_good) > 0:
                        if img_path not in processed_img and clus_count <= d and img_clus in pre_good:
                            processed_img.append((img_path))
                            image_cluster_count[img_clus] += 1
                            pre_good.remove(img_clus)
                            if image_cluster_count[img_clus] == cluster_size[img_clus]:
                                del image_cluster_count[img_clus]
                            flag = True
                            break
                    else:
                        if img_path not in processed_img and clus_count <= d and img_clus in pre_bad:
                            processed_img.append((img_path))
                            image_cluster_count[img_clus] += 1
                            pre_bad.remove(img_clus)
                            if image_cluster_count[img_clus] == cluster_size[img_clus]:
                                del image_cluster_count[img_clus]
                            flag = True
                            break
        if not flag:
            print("no valid image")

        if dataset == 'fashion':
            src_img = Image.fromarray(img_arrays[imgpath_to_idx[img_path]].astype(np.uint8))
        else:
            src_img = Image.open(img_path)

        ori_label = imgpath_orilabel_dict[img_path]

        source_yield_rate = (sources_with_new_pair + 1) / (total_sources_done + 2)
        pair_uniqueness_rate = (len(seen_unique) + 1) / (total_pairs_found + 2)

        if pair_uniqueness_rate < source_yield_rate:
            decay = 1.0 - (1.0 - source_yield_rate) * pair_uniqueness_rate
        else:
            decay = source_yield_rate

        discovery_rate = [hits_per_mr[m] / tries_per_mr[m] for m in range(5)]
        next_max = max(discovery_rate)
        order = sorted(range(5), key=lambda m: discovery_rate[m], reverse=True)
        remaining = list(order)

        k = 0
        any_violation = False
        found_new_pair_this_source = False

        while remaining and budget > 0:
            mr_idx = remaining.pop(0)

            follow_up_img = mr_5.test_mrs(src_img, mr_list2[mr_idx])

            if dataset == 'fashion':
                mr_key = base_path + img_path
            else:
                mr_key = img_path
            f_label = mr_mdata[mr_idx][mr_key][0]
            model_run_times += 1
            budget -= 1
            k += 1
            tries_per_mr[mr_idx] += 1

            violated = (f_label != ori_label)
            new_unique_this_step = False

            if violated:
                any_violation = True
                error_num += 1
                total_pairs_found += 1
                pair = (ori_label, f_label)
                if pair not in seen_unique:
                    seen_unique.add(pair)
                    hits_per_mr[mr_idx] += 1
                    found_new_pair_this_source = True
                    new_unique_this_step = True

                if img_path not in fault_record:
                    fault_record[img_path] = [(mr_list[mr_idx], ori_label, f_label)]
                else:
                    exists = any((ori == ori_label and f == f_label) for _, ori, f in fault_record[img_path])
                    if not exists:
                        fault_record[img_path].append((mr_list[mr_idx], ori_label, f_label))

            if img_path not in select_record:
                select_record[img_path] = [(mr_list[mr_idx], ori_label, f_label)]
            else:
                select_record[img_path].append((mr_list[mr_idx], ori_label, f_label))

            marginal_tries_at_k[k - 1] += 1
            if new_unique_this_step:
                marginal_unique_at_k[k - 1] += 1

            if not remaining:
                break

            next_k = k
            if next_k < max_k:
                marginal_rate = marginal_unique_at_k[next_k] / marginal_tries_at_k[next_k]
                if marginal_rate < source_yield_rate * 0.5:
                    break

            cur_max = max(hits_per_mr[m] / tries_per_mr[m] for m in remaining)
            if cur_max < next_max * decay:
                break

        if k > 0:
            if any_violation and image_cluster[img_path] in image_cluster_count:
                good_clus_list.append(image_cluster[img_path])
            elif not any_violation and image_cluster[img_path] in image_cluster_count:
                bad_clus_list.append(image_cluster[img_path])

        if found_new_pair_this_source:
            sources_with_new_pair += 1
        total_sources_done += 1

        flag_c = True
        for k, v in image_cluster_count.items():
            if v <= d:
                flag_c = False
        if flag_c:
            d += 1
            round += 1
            pre_good = good_clus_list
            pre_bad = bad_clus_list
            good_clus_list = []
            bad_clus_list = []

    trc = error_num / model_run_times
    print("TRC:", trc)
    all_pairs = []
    for v in fault_record.values():
        for e in v:
            all_pairs.append(e[1:])
    unique_pairs = remove_duplicates(all_pairs)
    if dataset == 'imagenet' or dataset == 'fruit360':
        faults = file_budget
    else:
        faults = 90
    fdr = len(unique_pairs) / faults
    print("FDR:", fdr)

    return select_record, trc, fdr


# CMPS++(-I2): Harmonic Mean (DeepGini, MaxP) + Euclidean 1-pick (ablation: remove MA2-A)
def CMPS_v2(ori_mdata, mr_mdata, budget):
    # clustering
    predictions = ori_mdata[2]
    if dataset == 'imagenet':
        n = 1000
    elif dataset == 'fruit360':
        n = 141
    else:
        n = 10
    predictions = np.array([np.pad(pred, (0, n - len(pred)), 'constant') for pred in predictions])
    scaler = StandardScaler()
    predictions_scaled = scaler.fit_transform(predictions)
    dbscan_labels = my_dbscan.dbscan(predictions_scaled, eps=0.15, min_samples=1)

    cluster_counts = Counter(dbscan_labels)
    cluster_size = {}
    for cluster_label, count in cluster_counts.items():
        cluster_size[cluster_label] = count
    image_cluster_count = {}
    for k in dbscan_labels:
        if k not in image_cluster_count:
            image_cluster_count[k] = 0

    imgpath_pred_dict = {}
    imgpath_orilabel_dict = {}
    image_uncertainty = dict()
    image_cluster = {}

    # Same Harmonic Mean as CMPS_extend
    for index in range(len(img_paths)):
        image_cluster[img_paths[index]] = dbscan_labels[index]
        imgpath_pred_dict[img_paths[index]] = predictions[index]
        ori_label = ori_mdata[1][index]
        imgpath_orilabel_dict[img_paths[index]] = ori_label

        gini = deepgini(predictions[index])
        mp = 1.0 - max(predictions[index])
        image_uncertainty[img_paths[index]] = 2 * gini * mp / (gini + mp + 1e-10)

    image_uncertainty_sorted = sorted(image_uncertainty.items(), key=lambda x: x[1], reverse=True)

    processed_img = []
    fault_record = {}
    select_record = {}
    model_run_times = 0
    error_num = 0
    d = 0

    good_clus_list = []
    bad_clus_list = []
    pre_good = []
    pre_bad = []
    round = 0
    file_budget = budget

    while budget > 0:

        budget -= 1
        flag = False

        for img_index in range(len(image_uncertainty_sorted)):

            img_path = image_uncertainty_sorted[img_index][0]
            img_clus = image_cluster[img_path]
            if img_clus in image_cluster_count:
                if round == 0:
                    clus_count = image_cluster_count[img_clus]
                    if img_path not in processed_img and clus_count <= d:
                        processed_img.append((img_path))
                        image_cluster_count[img_clus] += 1
                        if image_cluster_count[img_clus] == cluster_size[img_clus]:
                            del image_cluster_count[img_clus]
                        flag = True
                        break
                else:
                    clus_count = image_cluster_count[img_clus]
                    if len(pre_good) > 0:
                        if img_path not in processed_img and clus_count <= d and img_clus in pre_good:
                            processed_img.append((img_path))
                            image_cluster_count[img_clus] += 1
                            pre_good.remove(img_clus)
                            if image_cluster_count[img_clus] == cluster_size[img_clus]:
                                del image_cluster_count[img_clus]
                            flag = True
                            break
                    else:
                        if img_path not in processed_img and clus_count <= d and img_clus in pre_bad:
                            processed_img.append((img_path))
                            image_cluster_count[img_clus] += 1
                            pre_bad.remove(img_clus)
                            if image_cluster_count[img_clus] == cluster_size[img_clus]:
                                del image_cluster_count[img_clus]
                            flag = True
                            break
        if not flag:
            print("no valid image")

        # Euclidean 1-pick: select the MR with max Euclidean distance
        if dataset == 'fashion':
            ori_label = imgpath_orilabel_dict[img_path]
            eu_distance = []
            for mr in mr_list2:
                img = img_arrays[ori_mdata[0].index(img_path)]
                image_pil = Image.fromarray(img)
                followup_img = mr_5.test_mrs(image_pil, mr)
                followup_img_array = image.img_to_array(followup_img)
                euclidean_distance = euclidean(img.flatten(), followup_img_array.flatten())
                eu_distance.append(euclidean_distance)

            max_dis = max(eu_distance)
            max_index = eu_distance.index(max_dis)
            ind = base_path + img_path
            f_label = mr_mdata[max_index][ind][0]
            model_run_times += 1

            if ori_label != f_label:
                error_num += 1
                if image_cluster[img_path] in image_cluster_count:
                    good_clus_list.append(image_cluster[img_path])
                if img_path not in fault_record:
                    fault_record[img_path] = [(mr_list[max_index], ori_label, f_label)]
                else:
                    fault_record[img_path].append((mr_list[max_index], ori_label, f_label))
            else:
                if image_cluster[img_path] in image_cluster_count:
                    bad_clus_list.append(image_cluster[img_path])

            if img_path not in select_record:
                select_record[img_path] = [(mr_list[max_index], ori_label, f_label)]
            else:
                select_record[img_path].append((mr_list[max_index], ori_label, f_label))

        else:
            img = image.load_img(img_path)
            img = img.convert('RGB')
            ori_img_array = image.img_to_array(img)
            ori_label = imgpath_orilabel_dict[img_path]

            eu_distance = []
            for mr in mr_list2:
                followup_img = mr_5.test_mrs(img, mr)
                followup_img_array = image.img_to_array(followup_img)
                euclidean_distance = euclidean(ori_img_array.flatten(), followup_img_array.flatten())
                eu_distance.append(euclidean_distance)

            max_dis = max(eu_distance)
            max_index = eu_distance.index(max_dis)
            f_label = mr_mdata[max_index][img_path][0]
            model_run_times += 1

            if ori_label != f_label:
                error_num += 1
                if image_cluster[img_path] in image_cluster_count:
                    good_clus_list.append(image_cluster[img_path])
                if img_path not in fault_record:
                    fault_record[img_path] = [(mr_list[max_index], ori_label, f_label)]
                else:
                    fault_record[img_path].append((mr_list[max_index], ori_label, f_label))
            else:
                if image_cluster[img_path] in image_cluster_count:
                    bad_clus_list.append(image_cluster[img_path])

            if img_path not in select_record:
                select_record[img_path] = [(mr_list[max_index], ori_label, f_label)]
            else:
                select_record[img_path].append((mr_list[max_index], ori_label, f_label))

        flag_c = True
        for k, v in image_cluster_count.items():
            if v <= d:
                flag_c = False
        if flag_c:
            d += 1
            round += 1
            pre_good = good_clus_list
            pre_bad = bad_clus_list
            good_clus_list = []
            bad_clus_list = []

    trc = error_num / model_run_times
    print("TRC:", trc)
    all_pairs = []
    for v in fault_record.values():
        for e in v:
            all_pairs.append(e[1:])
    unique_pairs = remove_duplicates(all_pairs)
    if dataset == 'imagenet' or dataset == 'fruit360':
        faults = file_budget
    else:
        faults = 90
    fdr = len(unique_pairs) / faults
    print("FDR:", fdr)

    return select_record, trc, fdr


import csv

results = []

mr_list = ['flipLeftRight', 'gaussian', 'colored', 'rotatePlus5deg', 'brightness']
mr_list2 = ['flip_left_right 0', 'gaussian 2', 'colored 1.6', 'rotate 5', 'brightness 1.3']
cell_list = ['fileNamesArray', 'labelSourceArray', 'confidenceSourceArray', 'labelFollowUpArray', 'confidenceFollowUpArray']
dir_path = '/Users/miya_wang/Desktop/Papers/Second_paper/DNN-MT/matlab/'
datasets = ['fashion', 'cifar10', 'fruit360', 'imagenet']

for dataset in datasets:
    if dataset == 'fashion':
        base_path = '/Users/miya_wang/Desktop/CMPS++/Fashion-MNIST'
        for model in ['lenet1', 'lenet5']:
            print(dataset, model)
            path = dir_path + dataset + '_' + model + '/'
            ori_mdata, mr_mdata = data_process(path)

            img_arrays = np.load(base_path + '/fashion_test_images.npy')
            img_paths = ori_mdata[0]

            for budget in [500, 1000]:
                print(budget)
                print("CMPS++(-I1)")
                sr_v1, trc2, fdr2 = CMPS_v1(ori_mdata, mr_mdata, budget)
                save_v1_dir = '/Users/miya_wang/Desktop/CMPS++/CMPS_extended/Experiment_results/RQ3/CMPS++(-I1)/'
                os.makedirs(save_v1_dir, exist_ok=True)
                with open(os.path.join(save_v1_dir, f'{dataset}_{model}_{budget}.pkl'), 'wb') as fpkl:
                    pickle.dump(sr_v1, fpkl)
                print("CMPS++(-I2)")
                sr_v2, trc3, fdr3 = CMPS_v2(ori_mdata, mr_mdata, budget)
                save_v2_dir = '/Users/miya_wang/Desktop/CMPS++/CMPS_extended/Experiment_results/RQ3/CMPS++(-I2)/'
                os.makedirs(save_v2_dir, exist_ok=True)
                with open(os.path.join(save_v2_dir, f'{dataset}_{model}_{budget}.pkl'), 'wb') as fpkl:
                    pickle.dump(sr_v2, fpkl)
                results.append([dataset, model, budget, trc2, fdr2, trc3, fdr3])


    elif dataset == 'cifar10':
        base_path = '/Users/miya_wang/Desktop/CMPS++/cifar-10-batches-py/test_img/'
        for model in ['vgg19', 'resnet50']:
            print(dataset, model)
            path = dir_path + dataset + '_' + model + '/'
            ori_mdata, mr_mdata = data_process(path)

            image_names = ori_mdata[0]
            directory_path = base_path[:-1]
            img_paths = [os.path.join(directory_path, fname) for fname in image_names]
            if not img_paths:
                print("No images found in the specified directory.")
            else:
                img_arrays = load_and_preprocess_images(img_paths)

            for budget in [500, 1000]:
                print(budget)
                print("CMPS++(-I1)")
                sr_v1, trc2, fdr2 = CMPS_v1(ori_mdata, mr_mdata, budget)
                save_v1_dir = '/Users/miya_wang/Desktop/CMPS++/CMPS_extended/Experiment_results/RQ3/CMPS++(-I1)/'
                os.makedirs(save_v1_dir, exist_ok=True)
                with open(os.path.join(save_v1_dir, f'{dataset}_{model}_{budget}.pkl'), 'wb') as fpkl:
                    pickle.dump(sr_v1, fpkl)
                print("CMPS++(-I2)")
                sr_v2, trc3, fdr3 = CMPS_v2(ori_mdata, mr_mdata, budget)
                save_v2_dir = '/Users/miya_wang/Desktop/CMPS++/CMPS_extended/Experiment_results/RQ3/CMPS++(-I2)/'
                os.makedirs(save_v2_dir, exist_ok=True)
                with open(os.path.join(save_v2_dir, f'{dataset}_{model}_{budget}.pkl'), 'wb') as fpkl:
                    pickle.dump(sr_v2, fpkl)
                results.append([dataset, model, budget, trc2, fdr2, trc3, fdr3])


    elif dataset == "fruit360":
        base_path = '/Users/miya_wang/Desktop/CMPS++/Fruit360/fruit360_subset/'
        for model in ['mobilenetv2', 'shufflenet']:
            print(dataset, model)
            path = dir_path + dataset + '_' + model + '/'
            ori_mdata, mr_mdata = data_process(path)

            image_names = ori_mdata[0]
            directory_path = base_path[:-1]
            img_paths = [os.path.join(directory_path, fname) for fname in image_names]
            if not img_paths:
                print("No images found in the specified directory.")
            else:
                img_arrays = load_and_preprocess_images(img_paths)

            for budget in [500, 1000]:
                print(budget)
                print("CMPS++(-I1)")
                sr_v1, trc2, fdr2 = CMPS_v1(ori_mdata, mr_mdata, budget)
                save_v1_dir = '/Users/miya_wang/Desktop/CMPS++/CMPS_extended/Experiment_results/RQ3/CMPS++(-I1)/'
                os.makedirs(save_v1_dir, exist_ok=True)
                with open(os.path.join(save_v1_dir, f'{dataset}_{model}_{budget}.pkl'), 'wb') as fpkl:
                    pickle.dump(sr_v1, fpkl)
                print("CMPS++(-I2)")
                sr_v2, trc3, fdr3 = CMPS_v2(ori_mdata, mr_mdata, budget)
                save_v2_dir = '/Users/miya_wang/Desktop/CMPS++/CMPS_extended/Experiment_results/RQ3/CMPS++(-I2)/'
                os.makedirs(save_v2_dir, exist_ok=True)
                with open(os.path.join(save_v2_dir, f'{dataset}_{model}_{budget}.pkl'), 'wb') as fpkl:
                    pickle.dump(sr_v2, fpkl)
                results.append([dataset, model, budget, trc2, fdr2, trc3, fdr3])


    elif dataset == 'imagenet':
        base_path = '/Users/miya_wang/Desktop/CMPS++/ImageNet/'
        for model in ['googlenet', 'resnet50']:
            print(dataset, model)
            path = dir_path + dataset + '_' + model + '/'
            ori_mdata, mr_mdata = data_process(path)

            directory_path = base_path[:-1]
            image_names = ori_mdata[0]
            img_paths = [os.path.join(directory_path, fname) for fname in image_names]
            if not img_paths:
                print("No images found in the specified directory.")
            else:
                img_arrays = load_and_preprocess_images(img_paths)

            for budget in [500, 1000]:
                print(budget)
                print("CMPS++(-I1)")
                sr_v1, trc2, fdr2 = CMPS_v1(ori_mdata, mr_mdata, budget)
                save_v1_dir = '/Users/miya_wang/Desktop/CMPS++/CMPS_extended/Experiment_results/RQ3/CMPS++(-I1)/'
                os.makedirs(save_v1_dir, exist_ok=True)
                with open(os.path.join(save_v1_dir, f'{dataset}_{model}_{budget}.pkl'), 'wb') as fpkl:
                    pickle.dump(sr_v1, fpkl)
                print("CMPS++(-I2)")
                sr_v2, trc3, fdr3 = CMPS_v2(ori_mdata, mr_mdata, budget)
                save_v2_dir = '/Users/miya_wang/Desktop/CMPS++/CMPS_extended/Experiment_results/RQ3/CMPS++(-I2)/'
                os.makedirs(save_v2_dir, exist_ok=True)
                with open(os.path.join(save_v2_dir, f'{dataset}_{model}_{budget}.pkl'), 'wb') as fpkl:
                    pickle.dump(sr_v2, fpkl)
                results.append([dataset, model, budget, trc2, fdr2, trc3, fdr3])

# Write results to CSV
csv_path = '/Users/miya_wang/Desktop/CMPS++/CMPS_extended/Experiment_results/cmps_variants3_results.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Dataset', 'Model', 'Budget',
                     'CMPS++(-I1)_TRC', 'CMPS++(-I1)_FDR',
                     'CMPS++(-I2)_TRC', 'CMPS++(-I2)_FDR'])
    for row in results:
        writer.writerow(row)
print(f"\nResults saved to {csv_path}")
