"""Sensitivity analysis for the CMPS++ stopping-rule design choices.

Evaluates the marginal-return coefficient alpha in {0.25, 0.5, 0.75, 1.0}
with the proposed adaptive delta, plus a non-adaptive delta = SYR comparison.
The script follows the eight-subject, two-budget setup in cmps_variants.py and
writes only TRC and FDR results.
"""

import csv
import os

from cmps_variants_new import (
    CMPS_stopping_sensitivity,
    DIR_PATH,
    _set_dataset_inputs,
    _subjects,
    data_process,
)


OUTPUT_DIR = '/Users/miya_wang/Desktop/CMPS++/CMPS_extended/Experiment_results/RQ4'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'cmps_stopping_threshold_sensitivity.csv')


CONFIGURATIONS = [
    ('alpha=0.25', 0.25, 'adaptive'),
    ('alpha=0.50', 0.50, 'adaptive'),
    ('alpha=0.75', 0.75, 'adaptive'),
    ('alpha=1.00', 1.00, 'adaptive'),
    ('delta=SYR', 0.50, 'syr'),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    for dataset, model, base_path in _subjects():
        print(f'\n{dataset} / {model}')
        ori_mdata, mr_mdata = data_process(
            DIR_PATH + dataset + '_' + model + '/', base_path
        )
        img_paths, img_arrays = _set_dataset_inputs(dataset, base_path, ori_mdata)

        for budget in (500, 1000):
            row = [dataset, model, budget]
            for label, alpha, decay_mode in CONFIGURATIONS:
                _, trc, fdr = CMPS_stopping_sensitivity(
                    ori_mdata, mr_mdata, budget, dataset, base_path,
                    img_paths, img_arrays,
                    marginal_alpha=alpha, decay_mode=decay_mode,
                )
                print(f'  budget={budget}, {label}: TRC={trc:.6f}, FDR={fdr:.6f}')
                row.extend([trc, fdr])
            rows.append(row)

    header = ['Dataset', 'Model', 'Budget']
    for label, _, _ in CONFIGURATIONS:
        header.extend([f'{label}_TRC', f'{label}_FDR'])
    with open(OUTPUT_CSV, 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    print(f'\nResults saved to {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
