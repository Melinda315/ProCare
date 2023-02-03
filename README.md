# ProCare
This repository contains a demo implementation of our ProCare.

Please note that if the local pytorch contains the cpu version, it may cause an error to be reported, which is a library issue

## Environment Setup
1. Pytorch 1.12.1

2. Python 3.7.15

3. torch-scatter: pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.1+cu102.html

4. torch-sparse: pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.12.1+cu102.html

5. torch-geometric: pip install torch-geometric

6. torchdiffeq: pip install torchdiffeq

   

## Guideline

### data

We provide one implementation on MIMIC-III dataset.

`binary_train_codes_x.pkl` train set

`binary_test_codes_x.pkl` test set

`train_codes_y.npy` train label

`test_codes_y.npy` test label

`train_visit_lens.npy` the number of visits per patient in train set

`test_visit_lens.npy` the number of visits per patient in test set

`code_levels.npy` ICD-9 disease code tree structure

`patient_time_duration_encoded.pkl` timestamp of all patient visits

`sum_TE2.0.pkl` the severity-driven time embedding

`train_pids.npy` the mapping of patient's pids to idx in train set

`test_pids.npy` the mapping of patient's pids to idx in test set

### models

The implementation of model(```ProCare.py```); 

## Example to run the codes

```
python ProCare.py
```



