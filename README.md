# CMPS++: Uncertainty and Diversity Driven Metamorphic Test Case Pair Selection for Deep Neural Networks

This folder serves as the replication package for the paper: “CMPS++: Uncertainty and Diversity Driven Metamorphic Test Case Pair Selection for Deep Neural Networks”. It includes the source code of the proposed approach and baselines, all necessary scripts to reproduce the experiments, the data required for running the experiments, and the original experimental results.

---

## Folder Structure

```
.
├── Input_data
│   ├── Fault_clusters
│   ├── Pretrained_model
│   └── Retrain
├── Experiment_results
│   ├── RQ1
│   ├── RQ2&3
│   └── RQ4
├── Readme.md
├── requirements.txt
└── Source_code
```

## Requirements

The `requirements.txt` lists the dependencies required to run the Python code.
Run the following command to install dependency packages:

```
pip install -r requirements.txt
```

## CMPS++

The `Source_code\.py`file contains the implementation of  the **CMPS++** approach.
Run the following function to perform MP selection for a given test selection problem:

```python
def sets(size, index, features,output_probability,uncertainty,diversity,a)
```

The input parameters include:

* `size`: test budget (int)
* `index`: indexes of all the test inputs (list)
* `features`: features of all the test inputs (numpy)
* `output_probability`: the output probabilities of all the test inputs (numpy)
* `uncertainty`: "maxp" or "gini" (str)
* `diversity`: "gd" or "std" (str)
* `a`: the reduction coefficient (int)

The output will be  a list of indexes of the selected test inputs and the execution time.

## Reproducing Experiments

### 1) Experimental Subjects

In our experiments, we selected four datasets and assigned two DNN models to each dataset, i.e., a total of eight experimental subjects, as follows:

| Dataset       | Test Set | DNN Model    |
|:-------------:|:--------:|:-------------:|
| Fashion-MNIST | 10,000   | LeNet1        |
|               |          | LeNet5        |
| CIFAR-10      | 10,000   | VGG19         |
|               |          | ResNet50      |
| ImageNet      | 10,000   | GoogleNet     |
|               |          | ResNet50      |
| Fruit-360     | 10,000    | MobileNetV2   |
|               |          | ShuffleNet    |`

We provide all the datasets and pretrained DNN models in the `Input_data\Subjects` folder.

### 2) Metamorphic Relations (MRs)

We selected five different MRs:

* **MR1 - Flip Left-right** : The DNN model’s outputs should remain consistent when the image is flipped from left to right.
* **MR2 - Gaussian Blur** : The DNN model’s outputs should remain consistent when the image undergoes Gaussian blurring.
* **MR3 - Rotate 5°** : The DNN model’s outputs should remain consistent when the image is rotated by 5&deg;.
* **MR4 - Change Chromaticity** : The DNN model’s outputs should remain consistent when the chromaticity of the image is increased.
* **MR5 - Adjust Brightness** : The DNN model’s outputs should remain consistent when the brightness of the image is increased.

Here's an example demonstrating how MRs transform the source test case into its follow-up test cases.

![Examples of the Selected MRs Applied to an Image of a Cat](mrs_example.png)

The implementations of these five MRs are in the `mr_5.py` file under the `src` folder. You can also add new MRs based on your own needs.
