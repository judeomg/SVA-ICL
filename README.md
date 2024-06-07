# SVA-ICL: Improving LLM-based Software Vulnerability Assessment via In-Context Learning and Bimodal Information

This is the source code to the paper "SVA-ICL: Improving LLM-based Software Vulnerability Assessment via In-Context
Learning and Bimodal Information". Please refer to the paper for the experimental details.

# Approach



## About dataset.

1. The `dataset` folder contains all the data used in the experiments for RQ1-RQ5.
2. The `dataset2` and `dataset3` folders store the additional two random samples used in the discussion section.
3. Due to the large size of the datasets, we have stored them in Google Drive: [Google Drive Link](https://drive.google.com/drive/folders/1endc4u6qjaaOUJ0Sxks2PYGYeDC9mJ5p?usp=drive_link).

## About the experimental results in the paper:

1. The results for RQ1 and RQ2 are stored in the `results3` and `results2` folders, respectively.
2. The results for RQ3 and RQ4 are stored in the `results_RQ3` and `results_RQ4` folders, respectively.
3. The results for RQ5 are stored in the `results` folder.
4. The experimental results for the discussion section are stored in the `results_gpt35`, `results_gpt4o`, `results_dataset2`, and `results_dataset3` folders.

## About the models:

We use the `bert_whitening` trained models, which are stored in the `model`, `model_dataset2`, and `model_dataset3` folders.

## For reproducing the experiments:

1. Use the provided Jupyter files for data preprocessing.
2. Run `bert_whitening.py`. After running, we get the semantic vector library of the training set, kernel, and bias.
3. Run `ccgir.py` to get the most similar code fragments for the test set.
4. Run `search_info_form_code.ipynb` to get all the data required for the prompt template.
5. Run `deepseek.ipynb` to call the LLM and complete the vulnerability assessment task.