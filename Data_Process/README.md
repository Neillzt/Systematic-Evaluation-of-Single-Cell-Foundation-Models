## How to Use:

1. Ensure all datasets are correctly set up and paths are configured in the scripts.
2. Run the desired script(s) based on the task:
   - For Data split to train and validate for scFoundation version: `Split.py`
   - For few-shot for scFoundation version: `few_shot_split.py`
   
3. Use `Dataset_statistics.ipynb` to understand the format and distribution of data.
4. Use `predicition.ipynb` to generate visualizations and analyze the predictive distribution of data.

For further details, refer to the comments within each script.



## Dataset Overview

| **Dataset**   | **Train** | **Test** | **Type** | **Note** |
|----------------|-----------|----------|----------|----------|
| **Lung-Kim**   | 23,185    | 7,387    | 10     | Class Imbalance     |
| **Covid-19**   | 15,997    | 4,003    | 39     |                     |
| **Myeloid**    | 9,748     | 3,430    | 21     |                     |
| **hPancreas**  | 10,600    | 4,218    | 13     | Class Imbalance     |
| **Multiple Sclerosis**         | 7,844     | 13,468   | 18     | Out of Distribution |
