## Introduction

Generate an interpreter for TBCNN by Robin.

## Environment

- Python 3.x
- Tensorflow 2.4.0

## Step Instructions

1. Install the dependencies.

   ```
   cd src
   pip install -r requirements.txt
   ```

2. Download the dataset from https://sites.google.com/site/treebasedcnn/, and process the data to AST.

   ```
   cd script
   source process_data.sh
   ```

3. Train a code functionality classifier to be explained.

   ```
   source tbcnn_testing_script.sh
   ```

4. Generate perturbed examples. Put the training set to "Generate_Perturbed_Examples" folder to generate candidate perturbed examples, and copy the candidate perturbed examples to this folder.

5. Filter out perturbed examples whose prediction labels are different from the prediction labels of the corresponding original examples in the training set.

   ```
   cd explain
   python3 load_difdata.py \
   --train_path ../OJ104_pycparser_train_test_val/pycparser-buckets-train.pkl --test_path ../OJ104_pycparser_train_test_val/pycparser-buckets-val.pkl --batch_size 32 \
   --checkpoint_every 500 --cuda -1 --validating 1 \
   --tree_size_threshold_upper 1500 \
   --tree_size_threshold_lower 0 \
   --node_type_dim 100 --node_token_dim 100 \
   --node_type_vocabulary_path ../vocab/pycparser/node_type/type.txt \
   --token_vocabulary_path ../vocab/pycparser/node_token/token.txt --epochs 120 --parser pycparser \
   --node_init 2 --num_conv 4 --conv_output_dim 100
   ```

6. Generate a preliminary interpreter.

   ```
   python3 robin_robust.py \
   --train_path ../OJ104_pycparser_train_test_val/pycparser-buckets-train.pkl --test_path ../OJ104_pycparser_train_test_val/pycparser-buckets-val.pkl --batch_size 32 \
   --checkpoint_every 500 --cuda -1 --validating 1 \
   --tree_size_threshold_upper 1500 \
   --tree_size_threshold_lower 0 \
   --node_type_dim 100 --node_token_dim 100 \
   --node_type_vocabulary_path ../vocab/pycparser/node_type/type.txt \
   --token_vocabulary_path ../vocab/pycparser/node_token/token.txt --epochs 120 --parser pycparser \
   --node_init 2 --num_conv 4 --conv_output_dim 100
   ```

7. Optimize the preliminary interpreter.

   ```
   python3 robin_rerobust.py \
   --train_path ../OJ104_pycparser_train_test_val/pycparser-buckets-train.pkl --test_path ../OJ104_pycparser_train_test_val/pycparser-buckets-val.pkl --batch_size 32 \
   --checkpoint_every 500 --cuda -1 --validating 1 \
   --tree_size_threshold_upper 1500 \
   --tree_size_threshold_lower 0 \
   --node_type_dim 100 --node_token_dim 100 \
   --node_type_vocabulary_path ../vocab/pycparser/node_type/type.txt \
   --token_vocabulary_path ../vocab/pycparser/node_token/token.txt --epochs 120 --parser pycparser \
   --node_init 2 --num_conv 4 --conv_output_dim 100
   ```

   
