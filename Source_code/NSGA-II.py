from scipy.io import loadmat
import random
import pickle
import json

# mr1 = flipLeftRight
# mr2 = gaussian
# mr3 = rotatePlus30deg
# mr4 = colored
# mr5 = brightness


def remove_duplicates(pairs):

    unique_pairs = set()
    for pair in pairs:

        unique_pairs.add(tuple(pair))

    # 将集合转换回列表并返回
    return unique_pairs

def nsga(res_path,dir_path,dataset,model,budget,mr):

    mr_path = res_path + '/' + dataset + '_' + model + '/' + dataset + '_' + model + '_'+ str(mr) +'.mat'
    mat_data = loadmat(mr_path)
    str_key = 'selectArray_'+str(mr)
    selectfiles_mr = (mat_data[str_key][0]).tolist()
    selected_elements = random.sample(selectfiles_mr, budget)
    mr_name = mrs[str(mr)]
    file = loadmat(dir_path +dataset + '_' + model + '/' + mr_name + '_classification_results.mat')
    variable_data = file['fileNamesArray'][()]
    filenames_mr = [item[0][0] for item in variable_data]
    variable_data = file['labelSourceArray'][()]
    labelSourceArray_mr = [item[0][0] if item and item[0].size else ' ' for item in variable_data]
    variable_data = file['labelFollowUpArray'][()]
    labelFollowUpArray_mr = [item[0][0] if item and item[0].size else ' ' for item in variable_data]
    error_num = 0
    all_pairs = []
    for i in selected_elements:
        index = i-1 # since the list of selectfiles_mr in matlab start from 1, we have to -1 in python
        if labelSourceArray_mr[index] != labelFollowUpArray_mr[index]:
            error_num += 1
            all_pairs.append((labelSourceArray_mr[index], labelFollowUpArray_mr[index]))
    uniq_p = remove_duplicates(all_pairs)

    return error_num ,uniq_p,selected_elements

mrs = {'mr1':'flipLeftRight', 'mr2':'gaussian', 'mr3':'rotatePlus5deg', 'mr4':'colored', 'mr5':'brightness'}
res_path = '/Users/miya_wang/Desktop/CMPS_replication/results/NSGA-II' #you should change the path before you run
dir_path = '/Users/miya_wang/Desktop/Papers/Second_paper/DNN-MT/matlab/'
out_path = '/Users/miya_wang/Desktop/CMPS_replication/results/NSGA-II/results/'

for dataset in ['fashion','cifar10','imagenet']:
    if dataset == 'fashion':
        for model in ['lenet1','lenet5']:
            print(dataset,model)
            for budget in [500,1000]:
                print(budget)
                trc_list = []
                fdr_list = []
                all_round_selected = []
                for i in range(10):
                    errors_list = []
                    all_uniq_pairs = set()
                    selected_indices = []
                    for mr in mrs.keys():
                        errors, uniq_pairs,selected_elements = nsga(res_path, dir_path, dataset, model, int(budget/5), mr)
                        selected_indices.append(selected_elements)
                        errors_list.append(errors)
                        all_uniq_pairs |= uniq_pairs
                    all_round_selected.append(selected_indices)
                    trc = sum(errors_list)/budget
                    fdr = len(all_uniq_pairs)/90
                    trc_list.append(trc)
                    fdr_list.append(fdr)
                filename = out_path + dataset + '_' + model + '_' + str(budget) + '.pkl'
                with open(filename, 'w') as f:
                    json.dump(all_round_selected, f)
                print("TRC:",sum(trc_list)/10)
                print("FDR:", sum(fdr_list)/10)

    elif dataset == 'cifar10':
        for model in ['vgg19','resnet50']:
            print(dataset, model)
            for budget in [500, 1000]:
                print(budget)
                trc_list = []
                fdr_list = []
                all_round_selected = []
                for i in range(10):
                    errors_list = []
                    all_uniq_pairs = set()
                    selected_indices = []
                    for mr in mrs.keys():
                        errors, uniq_pairs,selected_elements = nsga(res_path, dir_path, dataset, model, int(budget / 5), mr)
                        selected_indices.append(selected_elements)
                        errors_list.append(errors)
                        all_uniq_pairs |= uniq_pairs
                    all_round_selected.append(selected_indices)
                    trc = sum(errors_list) / budget
                    fdr = len(all_uniq_pairs) / 90
                    trc_list.append(trc)
                    fdr_list.append(fdr)
                filename = out_path + dataset + '_' + model + '_' + str(budget) + '.pkl'
                with open(filename, 'w') as f:
                    json.dump(all_round_selected, f)
                print("TRC:", sum(trc_list) / 10)
                print("FDR:", sum(fdr_list) / 10)

    elif dataset == 'imagenet':
        for model in ['googlenet','resnet50']:
            print(dataset, model)
            for budget in [500,1000]:
                print(budget)
                trc_list = []
                fdr_list = []
                all_round_selected = []
                for i in range(10):
                    errors_list = []
                    all_uniq_pairs = set()
                    selected_indices = []
                    for mr in mrs.keys():
                        errors, uniq_pairs,selected_elements = nsga(res_path, dir_path, dataset, model, int(budget/5), mr)
                        selected_indices.append(selected_elements)
                        errors_list.append(errors)
                        all_uniq_pairs |= uniq_pairs
                    all_round_selected.append(selected_indices)
                    trc = sum(errors_list) / budget
                    fdr = len(all_uniq_pairs) / budget
                    trc_list.append(trc)
                    fdr_list.append(fdr)
                filename = out_path + dataset + '_' + model+'_'+str(budget) + '.pkl'
                with open(filename, 'w') as f:
                    json.dump(all_round_selected, f)
                print("TRC:", sum(trc_list) / 10)
                print("FDR:", sum(fdr_list) / 10)

