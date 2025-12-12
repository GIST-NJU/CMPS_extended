import pickle
import random
import scipy.io


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def remove_duplicates(pairs):
    # set can filter out the same fault type
    unique_pairs = set()
    for pair in pairs:
        unique_pairs.add(tuple(pair))
    return list(unique_pairs)

# random selection (RS) approach
def generate_unique_pairs(num_pairs):
    pairs = set()
    while len(pairs) < num_pairs:
        first_num = random.randint(0, 4) # mr
        second_num = random.randint(0, 9999) # test case
        pairs.add((first_num, second_num))
    return pairs

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
                    cell_data_list.append(filenames)
                elif variable_name == 'labelSourceArray':
                    labelSourceArray = [item[0][0] if item and item[0].size else ' ' for item in variable_data]
                    ori_mdata[1] = labelSourceArray
                elif variable_name == 'labelFollowUpArray':
                    filenames = [item[0][0] if item and item[0].size else ' ' for item in variable_data]
                    cell_data_list.append(filenames)
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
            image_path = cell_data_list[0][index1]
            cell_dict[image_path] = [cell_data_list[1][index1]]
            for index2 in range(2, len(cell_data_list)):
                cell_dict[image_path].append(cell_data_list[index2][index1])
        mr_mdata.append(cell_dict)
    return ori_mdata,mr_mdata

def random_selection(ori_mdata, mr_mdata,budget,faults):
    times =10 #repeat time
    total_uniq = 0
    total_error = 0
    all_results = []  # store the raw data of RS
    for i in range(times):
        unique_pairs = generate_unique_pairs(budget)
        all_results.append(list(unique_pairs))
        count = 0
        all_pairs = []
        for pair in unique_pairs:
            index1 = pair[0]  # mr
            index2 = pair[1]  # src
            img_n = ori_mdata[0][index2]
            img_p = img_n
            img_label = ori_mdata[1][index2]
            followup_label = mr_mdata[index1][img_p][0]
            # print(img_label, followup_label)
            if img_label != followup_label:
                count += 1
                all_pairs.append((img_label, followup_label))

        uniq_p = remove_duplicates(all_pairs)
        total_uniq += len(uniq_p)
        total_error += len(all_pairs)

    output_path = '/Users/miya_wang/Desktop/Papers/Second_paper/DNN-MT/replication/random/' + dataset + '_' + model + '_' + str(budget) + '.pkl'
    with open(output_path, "wb") as f:
        pickle.dump(all_results, f)

    print("TRC:",total_error / (budget * times))
    print("FDR:",total_uniq / (times*faults))




mr_list = ['flipLeftRight', 'gaussian', 'colored', 'rotatePlus5deg', 'brightness']
mr_list2 = ['flip_left_right 0', 'gaussian 2', 'colored 1.6', 'rotate 5', 'brightness 1.3']
cell_list = ['fileNamesArray', 'labelSourceArray', 'confidenceSourceArray', 'labelFollowUpArray', 'confidenceFollowUpArray']
dir_path = '/Users/miya_wang/Desktop/Papers/Second_paper/DNN-MT/matlab/' #you should change the path when running
datasets = ['fashion','cifar10','imagenet']
# start RS
for dataset in datasets:
    if dataset == 'fashion':
        for model in ['lenet1','lenet5']:
            print(dataset, model)
            path = dir_path + dataset + '_' + model + '/'
            ori_mdata, mr_mdata = data_process(path)
            for budget in [500,1000]:
                print(budget)
                faults = 90  # the total number of fault types
                random_selection(ori_mdata, mr_mdata, budget, faults)

    elif dataset == 'cifar10':
        for model in ['vgg19','resnet50']:
            print(dataset, model)
            path = dir_path + dataset + '_' + model + '/'
            ori_mdata, mr_mdata = data_process(path)
            for budget in [500,1000]:
                print(budget)
                faults = 90
                random_selection(ori_mdata, mr_mdata, budget, faults)

    elif dataset == 'imagenet':
        for model in ['googlenet','resnet50']:
            print(dataset, model)
            path = dir_path + dataset + '_' + model + '/'
            ori_mdata, mr_mdata = data_process(path)
            for budget in [500, 1000]:
                print(budget)
                faults = budget
                random_selection(ori_mdata, mr_mdata, budget, faults)
















