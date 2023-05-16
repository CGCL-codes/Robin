## Introduction

Generate perturbed examples from given code examples by conducting semantics-preserving code transformations, involving 23 types of transformations.

## Environment

- Ubuntu
- Srcml (https://www.srcml.org/)
- Python3.x

## Step Instructions

1. Put the code examples to be transformed in "./program_file/test" directory.

2. Generate the XML file of the code examples.

   ```
   python get_style.py
   ```

3. Generate perturbed examples by applying code transformations.

   ```
   python3 untargeted_attack.py --form=generate
   ```
