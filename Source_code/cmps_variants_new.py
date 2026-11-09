"""Additional stopping-mechanism ablations for CMPS++.

This script keeps the experiment settings of ``cmps_variants.py`` (datasets,
models, MRs, budgets, clustering, source scheduling, and harmonic-mean
uncertainty) and adds three variants required to validate the revised
stopping mechanism:

* Original-stop: the perturbation-ordered stopping mechanism from
  ``cmps_extended_old.py``, with only its uncertainty score replaced by the
  current harmonic mean of DeepGini and MaxP.
* -Marginal: disables only MarginalRate < SYR / 2; the DR--delta rule remains.
* -Decay: disables only the DR--delta rule; the MarginalRate rule remains.

The output pickle format is the same as the existing RQ3 variants: a mapping
from source path to the executed (MR, source-label, follow-up-label) tuples.
"""

import csv
import os
import pickle
from collections import Counter

import numpy as np
import scipy.io
from PIL import Image
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing import image

import mr_5
import my_dbscan


MR_LIST = ['flipLeftRight', 'gaussian', 'colored', 'rotatePlus5deg', 'brightness']
MR_LIST2 = ['flip_left_right 0', 'gaussian 2', 'colored 1.6', 'rotate 5', 'brightness 1.3']
CELL_LIST = ['fileNamesArray', 'labelSourceArray', 'confidenceSourceArray',
             'labelFollowUpArray', 'confidenceFollowUpArray']
DIR_PATH = '/Users/miya_wang/Desktop/Papers/Second_paper/DNN-MT/matlab/'
RESULT_ROOT = '/Users/miya_wang/Desktop/CMPS++/CMPS_extended/Experiment_results/RQ3'


def load_and_preprocess_images(img_paths):
    """Retained for consistency with cmps_variants.py's experiment setup."""
    arrays = []
    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(224, 224))
        arr = image.img_to_array(img)
        arrays.append(np.expand_dims(arr, axis=0))
    return np.vstack(arrays)


def data_process(path, base_path):
    """Load the precomputed MATLAB predictions used by all RQ3 variants."""
    ori_mdata = [None, None, None]
    mr_mdata = []
    for mr in MR_LIST:
        mat = scipy.io.loadmat(str(path) + mr + '_classification_results.mat')
        cell_data_list = []
        cell_dict = {}
        for cell in CELL_LIST:
            if cell not in mat:
                raise KeyError(f"Variable '{cell}' not found in {path}{mr}_classification_results.mat")
            values = mat[cell][()]
            if cell == 'confidenceSourceArray':
                ori_mdata[2] = [list(item[0][0]) for item in values]
            elif cell == 'confidenceFollowUpArray':
                cell_data_list.append([list(item[0][0]) for item in values])
            elif cell == 'labelSourceArray':
                ori_mdata[1] = [item[0][0] if item is not None and item[0].size > 0 else ' '
                                for item in values]
            elif cell == 'labelFollowUpArray':
                cell_data_list.append([item[0][0] if item is not None and item[0].size > 0 else ' '
                                       for item in values])
            else:
                ori_mdata[0] = [item[0][0] for item in values]
                cell_data_list.append(ori_mdata[0])
        for i, filename in enumerate(cell_data_list[0]):
            image_path = base_path + filename
            cell_dict[image_path] = [cell_data_list[1][i]]
            for data in cell_data_list[2:]:
                cell_dict[image_path].append(data[i])
        mr_mdata.append(cell_dict)
    return ori_mdata, mr_mdata


def deepgini(outputs):
    return 1.0 - sum(probability ** 2 for probability in outputs if probability > 0)


def harmonic_uncertainty(outputs):
    gini = deepgini(outputs)
    maxp_uncertainty = 1.0 - max(outputs)
    return 2.0 * gini * maxp_uncertainty / (gini + maxp_uncertainty + 1e-10)


def _class_count(dataset):
    if dataset == 'imagenet':
        return 1000
    if dataset == 'fruit360':
        return 141
    return 10


def _prepare_state(ori_mdata, dataset, img_paths):
    """Prepare the common, current-CMPS++ clustering and uncertainty state."""
    n_classes = _class_count(dataset)
    predictions = np.array([
        np.pad(prediction, (0, n_classes - len(prediction)), 'constant')
        for prediction in ori_mdata[2]
    ])
    scaled_predictions = StandardScaler().fit_transform(predictions)
    labels = my_dbscan.dbscan(scaled_predictions, eps=0.15, min_samples=1)
    cluster_size = dict(Counter(labels))
    return {
        'predictions': predictions,
        'image_cluster': dict(zip(img_paths, labels)),
        'image_uncertainty': {
            img_paths[i]: harmonic_uncertainty(predictions[i])
            for i in range(len(img_paths))
        },
        'cluster_size': cluster_size,
        'image_cluster_count': {cluster: 0 for cluster in cluster_size},
        'source_labels': dict(zip(img_paths, ori_mdata[1])),
        'sorted_images': None,
    }


def _select_source(state, processed, round_index, depth, pre_good, pre_bad):
    """Replicate the source/cluster scheduling in cmps_variants.py."""
    for img_path in state['sorted_images']:
        cluster = state['image_cluster'][img_path]
        counts = state['image_cluster_count']
        if cluster not in counts or img_path in processed or counts[cluster] > depth:
            continue
        if round_index != 0 and ((pre_good and cluster not in pre_good) or
                                 (not pre_good and cluster not in pre_bad)):
            continue
        processed.append(img_path)
        counts[cluster] += 1
        if round_index != 0 and pre_good:
            pre_good.remove(cluster)
        elif round_index != 0:
            pre_bad.remove(cluster)
        if counts[cluster] == state['cluster_size'][cluster]:
            del counts[cluster]
        return img_path
    raise RuntimeError('No valid source image remains before the test budget is exhausted.')


def _advance_round(state, depth, good_clusters, bad_clusters):
    if all(count > depth for count in state['image_cluster_count'].values()):
        return depth + 1, good_clusters, bad_clusters, [], []
    return depth, None, None, good_clusters, bad_clusters


def _source_image(dataset, img_path, imgpath_to_idx, img_arrays):
    if dataset == 'fashion':
        return Image.fromarray(img_arrays[imgpath_to_idx[img_path]].astype(np.uint8))
    return Image.open(img_path)


def _follow_up_label(dataset, base_path, img_path, mr_idx, mr_mdata):
    key = base_path + img_path if dataset == 'fashion' else img_path
    return mr_mdata[mr_idx][key][0]


def _record(selection, img_path, mr_idx, source_label, follow_up_label):
    selection.setdefault(img_path, []).append(
        (MR_LIST[mr_idx], source_label, follow_up_label)
    )


def _metrics(selection, budget, dataset):
    entries = [entry for entries in selection.values() for entry in entries]
    failures = [entry for entry in entries if entry[1] != entry[2]]
    unique_pairs = {(entry[1], entry[2]) for entry in failures}
    denominator = budget if dataset in ('imagenet', 'fruit360') else 90
    trc = len(failures) / len(entries) if entries else 0.0
    fdr = len(unique_pairs) / denominator
    return trc, fdr


def _run_feedback_variant(ori_mdata, mr_mdata, budget, dataset, base_path,
                          img_paths, img_arrays, use_marginal, use_decay,
                          marginal_alpha=0.5, decay_mode='adaptive'):
    """Run current CMPS++ with either one or both stopping rules enabled."""
    state = _prepare_state(ori_mdata, dataset, img_paths)
    state['sorted_images'] = [path for path, _ in sorted(
        state['image_uncertainty'].items(), key=lambda item: item[1], reverse=True)]
    imgpath_to_idx = {path: i for i, path in enumerate(img_paths)}
    processed, selection = [], {}
    good_clusters, bad_clusters, pre_good, pre_bad = [], [], [], []
    depth, round_index = 0, 0
    hits, tries = [1.0] * 5, [5.0] * 5
    marginal_hits, marginal_tries = [1.0] * 5, [2.0] * 5
    seen_pairs = set()
    sources_with_new_pair = total_sources_done = total_pairs_found = 0
    remaining_budget = budget

    while remaining_budget > 0:
        img_path = _select_source(state, processed, round_index, depth, pre_good, pre_bad)
        source_image = _source_image(dataset, img_path, imgpath_to_idx, img_arrays)
        source_label = state['source_labels'][img_path]
        syr = (sources_with_new_pair + 1) / (total_sources_done + 2)
        pur = (len(seen_pairs) + 1) / (total_pairs_found + 2)
        if decay_mode == 'adaptive':
            decay = 1.0 - (1.0 - syr) * pur if pur < syr else syr
        elif decay_mode == 'syr':
            # Non-adaptive comparison: retain DR stopping but remove PUR.
            decay = syr
        else:
            raise ValueError("decay_mode must be 'adaptive' or 'syr'")
        rates = [hits[i] / tries[i] for i in range(5)]
        initial_max_rate = max(rates)
        remaining = sorted(range(5), key=lambda i: rates[i], reverse=True)
        position = 0
        any_violation = found_new_pair = False

        while remaining and remaining_budget > 0:
            mr_idx = remaining.pop(0)
            # Preserve the transformation/validity-workload of CMPS++.
            mr_5.test_mrs(source_image, MR_LIST2[mr_idx])
            follow_up_label = _follow_up_label(dataset, base_path, img_path, mr_idx, mr_mdata)
            remaining_budget -= 1
            position += 1
            tries[mr_idx] += 1
            is_new_pair = False
            if follow_up_label != source_label:
                any_violation = True
                total_pairs_found += 1
                pair = (source_label, follow_up_label)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    hits[mr_idx] += 1
                    found_new_pair = is_new_pair = True
            _record(selection, img_path, mr_idx, source_label, follow_up_label)
            marginal_tries[position - 1] += 1
            if is_new_pair:
                marginal_hits[position - 1] += 1
            if not remaining:
                break
            if use_marginal and position < 5:
                next_marginal_rate = marginal_hits[position] / marginal_tries[position]
                if next_marginal_rate < syr * marginal_alpha:
                    break
            if use_decay:
                remaining_max_rate = max(hits[i] / tries[i] for i in remaining)
                if remaining_max_rate < initial_max_rate * decay:
                    break

        cluster = state['image_cluster'][img_path]
        if position > 0 and cluster in state['image_cluster_count']:
            (good_clusters if any_violation else bad_clusters).append(cluster)
        if found_new_pair:
            sources_with_new_pair += 1
        total_sources_done += 1
        new_depth, new_pre_good, new_pre_bad, good_clusters, bad_clusters = _advance_round(
            state, depth, good_clusters, bad_clusters)
        if new_depth != depth:
            depth, round_index = new_depth, round_index + 1
            pre_good, pre_bad = new_pre_good, new_pre_bad

    return selection, *_metrics(selection, budget, dataset)


def CMPS_minus_marginal(ori_mdata, mr_mdata, budget, dataset, base_path, img_paths, img_arrays):
    """CMPS++(-Marginal): retain only the DR--delta stopping condition."""
    return _run_feedback_variant(ori_mdata, mr_mdata, budget, dataset, base_path,
                                 img_paths, img_arrays, use_marginal=False, use_decay=True)


def CMPS_minus_decay(ori_mdata, mr_mdata, budget, dataset, base_path, img_paths, img_arrays):
    """CMPS++(-Decay): retain only the MarginalRate--SYR/2 condition."""
    return _run_feedback_variant(ori_mdata, mr_mdata, budget, dataset, base_path,
                                 img_paths, img_arrays, use_marginal=True, use_decay=False)


def CMPS_stopping_sensitivity(ori_mdata, mr_mdata, budget, dataset, base_path,
                              img_paths, img_arrays, marginal_alpha=0.5,
                              decay_mode='adaptive'):
    """Evaluate a stopping-threshold configuration with both rules enabled.

    ``marginal_alpha`` implements
    MarginalRate(i+1) < marginal_alpha * SYR.  ``decay_mode='syr'`` uses the
    non-adaptive comparison delta = SYR; ``'adaptive'`` uses the proposed
    PUR-aware piecewise delta.
    """
    return _run_feedback_variant(
        ori_mdata, mr_mdata, budget, dataset, base_path, img_paths, img_arrays,
        use_marginal=True, use_decay=True,
        marginal_alpha=marginal_alpha, decay_mode=decay_mode,
    )


def CMPS_original_stop(ori_mdata, mr_mdata, budget, dataset, base_path, img_paths, img_arrays):
    """Previous perturbation-monotonic stopping rule with current uncertainty.

    This is ``cmps_extended_old.CMPS_extend`` with its former weighted score
    replaced by the current harmonic uncertainty.  Its MR ordering and stopping
    behaviour intentionally remain unchanged: descending pixel-space Euclidean
    perturbation; stop on the first non-violation or a repeated label pair.
    """
    state = _prepare_state(ori_mdata, dataset, img_paths)
    state['sorted_images'] = [path for path, _ in sorted(
        state['image_uncertainty'].items(), key=lambda item: item[1], reverse=True)]
    imgpath_to_idx = {path: i for i, path in enumerate(img_paths)}
    processed, selection = [], {}
    good_clusters, bad_clusters, pre_good, pre_bad = [], [], [], []
    depth, round_index = 0, 0
    remaining_budget = budget

    while remaining_budget > 0:
        img_path = _select_source(state, processed, round_index, depth, pre_good, pre_bad)
        source_image = _source_image(dataset, img_path, imgpath_to_idx, img_arrays)
        source_label = state['source_labels'][img_path]
        source_array = image.img_to_array(source_image)
        distances = []
        for mr_spec in MR_LIST2:
            follow_up = mr_5.test_mrs(source_image, mr_spec)
            distances.append(euclidean(source_array.flatten(), image.img_to_array(follow_up).flatten()))
        order = sorted(range(5), key=lambda i: distances[i], reverse=True)
        local_pairs, first_result = set(), None

        for position, mr_idx in enumerate(order):
            if remaining_budget <= 0:
                break
            follow_up_label = _follow_up_label(dataset, base_path, img_path, mr_idx, mr_mdata)
            remaining_budget -= 1
            violated = follow_up_label != source_label
            if first_result is None:
                first_result = violated
            _record(selection, img_path, mr_idx, source_label, follow_up_label)
            if not violated:
                break
            pair = (source_label, follow_up_label)
            if pair in local_pairs:
                break
            local_pairs.add(pair)

        cluster = state['image_cluster'][img_path]
        if first_result is not None and cluster in state['image_cluster_count']:
            (good_clusters if first_result else bad_clusters).append(cluster)
        new_depth, new_pre_good, new_pre_bad, good_clusters, bad_clusters = _advance_round(
            state, depth, good_clusters, bad_clusters)
        if new_depth != depth:
            depth, round_index = new_depth, round_index + 1
            pre_good, pre_bad = new_pre_good, new_pre_bad

    return selection, *_metrics(selection, budget, dataset)


def _subjects():
    return [
        ('fashion', 'lenet1', '/Users/miya_wang/Desktop/CMPS++/Fashion-MNIST'),
        ('fashion', 'lenet5', '/Users/miya_wang/Desktop/CMPS++/Fashion-MNIST'),
        ('cifar10', 'vgg19', '/Users/miya_wang/Desktop/CMPS++/cifar-10-batches-py/test_img/'),
        ('cifar10', 'resnet50', '/Users/miya_wang/Desktop/CMPS++/cifar-10-batches-py/test_img/'),
        ('fruit360', 'mobilenetv2', '/Users/miya_wang/Desktop/CMPS++/Fruit360/fruit360_subset/'),
        ('fruit360', 'shufflenet', '/Users/miya_wang/Desktop/CMPS++/Fruit360/fruit360_subset/'),
        ('imagenet', 'googlenet', '/Users/miya_wang/Desktop/CMPS++/ImageNet/'),
        ('imagenet', 'resnet50', '/Users/miya_wang/Desktop/CMPS++/ImageNet/'),
    ]


def _set_dataset_inputs(dataset, base_path, ori_mdata):
    if dataset == 'fashion':
        return ori_mdata[0], np.load(base_path + '/fashion_test_images.npy')
    directory = base_path[:-1]
    img_paths = [os.path.join(directory, name) for name in ori_mdata[0]]
    # Non-Fashion variants load only the selected source image, as does the
    # selection code itself.  Preloading every ImageNet image is unnecessary
    # for these experiments and can exhaust memory without affecting results.
    return img_paths, None


def main():
    variants = [
        ('CMPS++(Original-stop)', CMPS_original_stop),
        ('CMPS++(-Marginal)', CMPS_minus_marginal),
        ('CMPS++(-Decay)', CMPS_minus_decay),
    ]
    rows = []
    for dataset, model, base_path in _subjects():
        print(f'\n{dataset} / {model}')
        ori_mdata, mr_mdata = data_process(DIR_PATH + dataset + '_' + model + '/', base_path)
        img_paths, img_arrays = _set_dataset_inputs(dataset, base_path, ori_mdata)
        for budget in (500, 1000):
            result_row = [dataset, model, budget]
            for variant_name, function in variants:
                print(f'  {variant_name}, budget={budget}')
                selected, trc, fdr = function(
                    ori_mdata, mr_mdata, budget, dataset, base_path, img_paths, img_arrays
                )
                output_dir = os.path.join(RESULT_ROOT, variant_name)
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, f'{dataset}_{model}_{budget}.pkl'), 'wb') as stream:
                    pickle.dump(selected, stream)
                print(f'    TRC={trc:.6f}, FDR={fdr:.6f}')
                result_row.extend([trc, fdr])
            rows.append(result_row)

    csv_path = os.path.join(RESULT_ROOT, 'cmps_stopping_rule_ablation_results.csv')
    with open(csv_path, 'w', newline='') as stream:
        writer = csv.writer(stream)
        header = ['Dataset', 'Model', 'Budget']
        for variant_name, _ in variants:
            header.extend([f'{variant_name}_TRC', f'{variant_name}_FDR'])
        writer.writerow(header)
        writer.writerows(rows)
    print(f'\nResults saved to {csv_path}')


if __name__ == '__main__':
    main()
