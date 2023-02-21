## Introduction

Generate an interpreter on DL-CAIS by Robin.

## Environment

- Python 3.x
- Tensorflow 1.12.0
- Keras 2.2.4

## Step Instructions

1. Install the dependencies.

   ```
   cd src
   pip install -r requirements.txt
   ```

2. Download the dataset from https://github.com/EQuiw/code-imitator.

3. Train a code authorship attribution classifier to be explained.

   ```
   python evaluations/learning/rf_usenix/train_models_parallel.py
   ```

4. Generate perturbed example. Put the training set to "Generate_Perturbed_Examples" folder to generate candidate perturbed examples, and copy the candidate perturbed examples to this folder.

5. Filter out perturbed examples whose prediction labels are different from the prediction labels of the corresponding original examples in the training set.

   ```
   cd explain
   python get_nochange.py
   ```

6. Generate a preliminary interpreter.

   ```
   python train204_robust.py
   ```

7. Optimize the preliminary interpreter.

   ```
   python train204_rerobust.py
   ```

   
