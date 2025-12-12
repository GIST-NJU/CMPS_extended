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
    ori_mdata = [None, None,None]  # ori_data[0] = filenames, ori_data[1] = source_labels,ori_data[2] = output probabilities of source
    mr_mdata = []  # mr_mdata[0] = filenames, mr_mdata[1] = follow-up_labels, mr_mdata[2] = output probabilities of follow-ups
    for mr in mr_list:
        file = scipy.io.loadmat(str(path) + mr + '_classification_results.mat')
        cell_data_list = []
        cell_dict = {}
        for cell in cell_list:
            variable_name = cell
            if variable_name in file:
                variable_data = file[variable_name][()]
                # print(variable_data)
                # print(len(variable_data))
                if variable_name == 'confidenceSourceArray':
                    confidenceSourceArray = []
                    for item in variable_data:
                        confidenceSourceArray.append(list(item[0][0]))
                    ori_mdata[2] = confidenceSourceArray
                elif variable_name == 'confidenceFollowUpArray':
                    filenames = []
                    for item in variable_data:
                        filenames.append(list(item[0][0]))
                    cell_data_list.append(filenames) #cell_data_list[2] = 'confidenceFollowUpArray'
                elif variable_name == 'labelSourceArray':
                    labelSourceArray = [item[0][0] if item and item[0].size else ' ' for item in variable_data]
                    ori_mdata[1] = labelSourceArray
                elif variable_name == 'labelFollowUpArray':
                    filenames = [item[0][0] if item and item[0].size else ' ' for item in variable_data]
                    cell_data_list.append(filenames) #cell_data_list[1] = 'labelFollowUpArray'
                else:
                    filenames = [item[0][0] for item in variable_data]
                    ori_mdata[0] = filenames
                    cell_data_list.append(filenames)
                # print(filenames)
                # print(len(filenames))
                # print(variable_data)

            else:
                print(f"Variable '{variable_name}' not found in the .mat file.")
        for index1 in range(len(cell_data_list[0])):
            image_path = base_path + cell_data_list[0][index1]
            cell_dict[image_path] = [cell_data_list[1][index1]]
            for index2 in range(2, len(cell_data_list)):
                cell_dict[image_path].append(cell_data_list[index2][index1])
        mr_mdata.append(cell_dict)
    return ori_mdata,mr_mdata

def CMPS(ori_mdata,mr_mdata,budget):
    # clustering
    predictions = ori_mdata[2]
    if dataset == 'imagenet':
        n=1000
    else:
        n=10
    predictions = np.array([np.pad(pred, (0, n - len(pred)), 'constant') for pred in predictions])
    scaler = StandardScaler()
    predictions_scaled = scaler.fit_transform(predictions)
    dbscan_labels = my_dbscan.dbscan(predictions_scaled, eps=0.15, min_samples=1)

    cluster_counts = Counter(dbscan_labels)
    cluster_size = {}  # counter the number of each cluster
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
    for index in range(len(img_paths)):
        image_cluster[img_paths[index]] = dbscan_labels[index]  # image_path-number of clusters
        imgpath_pred_dict[img_paths[index]] = predictions[index]
        ori_label = ori_mdata[1][index]
        imgpath_orilabel_dict[img_paths[index]] = ori_label
        sum = 0
        for i in predictions[index]:
            if i > 0:
                sum += pow(i, 2)
        uncertainty = 1 - sum #get the uncertainty score
        image_uncertainty[img_paths[index]] = uncertainty

    image_uncertainty_sorted = sorted(image_uncertainty.items(), key=lambda x: x[1], reverse=True)  # rank based on uncertainty
    origin_sorted = image_uncertainty_sorted

    # start selection
    processed_img = []
    fault_record = {}
    model_run_times = 0
    error_num = 0
    d = 0

    good_clus_list = []  # record the good clusters of current iteration
    bad_clus_list = []  # record the bad clusters of current iteration
    pre_good = []  # store the good clusters of previous iteration
    pre_bad = []  # store the bad clusters of previous iteration
    round = 0  # update the iteration
    file_budget = budget

    while budget > 0:

        budget -= 1
        #fault_set = set()
        flag = False

        for img_index in range(len(image_uncertainty_sorted)):

            img_path = image_uncertainty_sorted[img_index][0]
            img_clus = image_cluster[img_path]
            if img_clus in image_cluster_count:
                if round == 0:  # the first iteration
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
                    else:  # 穷尽不出错的聚类
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
            ori_label = imgpath_orilabel_dict[img_path]

            fault_result = {}
            followup_img_array_list = []
            eu_distance = []
            for mr in mr_list2:
                img = img_arrays[ori_mdata[0].index(img_path)]
                image_pil = Image.fromarray(img)
                followup_img = mr_5.test_mrs(image_pil, mr) # the valid has been checked by SSIM and manual analysis in our experiment, you can directly compute the euclidean_distance
                followup_img_array = image.img_to_array(followup_img)
                followup_img_array_list.append(followup_img_array)

                euclidean_distance = euclidean(img.flatten(), followup_img_array.flatten())
                eu_distance.append(euclidean_distance)

            max_dis = max(eu_distance)
            max_index = eu_distance.index(max_dis)
            ind = base_path + img_path
            f_label = mr_mdata[max_index][ind][0]
            if ori_label == f_label:  # pass become bad_cluster
                model_run_times += 1
                if image_cluster[img_path] in image_cluster_count:
                    bad_clus_list.append(image_cluster[img_path])
            else:
                if image_cluster[img_path] in image_cluster_count:
                    good_clus_list.append(image_cluster[img_path])  # fail, become good_cluster

                model_run_times += 1
                error_num += 1
                if img_path not in fault_record:
                    fault_record[img_path] = [(mr_list[max_index], ori_label, f_label)]
                else:
                    fault_record[img_path].append((mr_list[max_index], ori_label, f_label))

        else: # cifar10 & imagenet
            img = image.load_img(img_path)
            img = img.convert('RGB')
            ori_img_array = image.img_to_array(img)
            ori_label = imgpath_orilabel_dict[img_path]

            fault_result = {}
            followup_img_array_list = []
            eu_distance = []

            for mr in mr_list2:
                followup_img = mr_5.test_mrs(img, mr)
                followup_img_array = image.img_to_array(followup_img)
                followup_img_array_list.append(followup_img_array)
                euclidean_distance = euclidean(ori_img_array.flatten(), followup_img_array.flatten())
                eu_distance.append(euclidean_distance)
            max_dis = max(eu_distance)
            max_index = eu_distance.index(max_dis)
            f_label = mr_mdata[max_index][img_path][0]
            if ori_label == f_label:
                model_run_times += 1
                if image_cluster[img_path] in image_cluster_count:
                    bad_clus_list.append(image_cluster[img_path])
            else:

                if image_cluster[img_path] in image_cluster_count:
                    good_clus_list.append(image_cluster[img_path])
                model_run_times += 1
                error_num += 1
                if img_path not in fault_record:
                    fault_record[img_path] = [(mr_list[max_index], ori_label, f_label)]
                else:
                    fault_record[img_path].append((mr_list[max_index], ori_label, f_label))

        flag_c = True
        for k, v in image_cluster_count.items():
            if v <= d:
                flag_c = False
        if flag_c:
            d += 1
            round += 1  # the new iteration
            pre_good = good_clus_list
            pre_bad = bad_clus_list
            good_clus_list = []
            bad_clus_list = []

    output_path = '/Users/miya_wang/Desktop/Papers/Second_paper/DNN-MT/replication/CMPS/' + dataset + '_' + model + '_' + str(file_budget) + '.pkl'
    with open(output_path,'w') as file:
        json.dump(fault_record, file)


    print("TRC:",error_num / model_run_times)

    all_pairs = []
    sum = 0
    for v in fault_record.values():
        for e in v:
            all_pairs.append(e[1:])
        sum += len(v)
    unique_pairs = remove_duplicates(all_pairs)
    if dataset == 'imagenet':
        faults = file_budget
    else:
        faults = 90

    print("FDR:", len(unique_pairs) / faults)





mr_list = ['flipLeftRight', 'gaussian', 'colored', 'rotatePlus5deg', 'brightness']
mr_list2 = ['flip_left_right 0', 'gaussian 2', 'colored 1.6', 'rotate 5', 'brightness 1.3']
cell_list = ['fileNamesArray', 'labelSourceArray', 'confidenceSourceArray', 'labelFollowUpArray', 'confidenceFollowUpArray']
dir_path = '/Users/miya_wang/Desktop/Papers/Second_paper/DNN-MT/matlab/' #you should change the path when running
datasets = ['fashion', 'cifar10', 'imagenet']

for dataset in datasets:
    if dataset =='fashion':
        base_path = '/Users/miya_wang/Desktop/CMPS_replication/subjects/datasets/Fashion-MNIST' #you should change the path when running
        for model in ['lenet1','lenet5']:
            print(dataset, model)
            path = dir_path + dataset + '_' + model + '/'
            ori_mdata, mr_mdata = data_process(path)

            img_arrays = np.load(base_path+'/fashion_test_images.npy')
            img_paths = ori_mdata[0]

            for budget in [500,1000]:
                print(budget)
                CMPS(ori_mdata, mr_mdata, budget)


    elif dataset =='cifar10':
        base_path = '/Users/miya_wang/Desktop/CMPS_replication/subjects/datasets/Cifar-10/test_img/'#you should change the path when running
        for model in ['vgg19','resnet50']:
            print(dataset, model)
            path = dir_path + dataset + '_' + model + '/'
            ori_mdata, mr_mdata = data_process(path)

            # get the image
            image_names = ori_mdata[0]
            directory_path = base_path[:-1]
            img_paths = [os.path.join(directory_path, fname) for fname in image_names]
            if not img_paths:
                print("No images found in the specified directory.")
            else:
                img_arrays = load_and_preprocess_images(img_paths)

            for budget in [500,1000]:
                print(budget)
                CMPS(ori_mdata, mr_mdata, budget)

    elif dataset =='imagenet':
        base_path = '/Users/miya_wang/Desktop/CMPS_replication/subjects/datasets/ImageNet/'#you should change the path when running
        for model in ['googlenet','resnet50']:
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

            for budget in [500,1000]:
                print(budget)
                CMPS(ori_mdata, mr_mdata, budget)