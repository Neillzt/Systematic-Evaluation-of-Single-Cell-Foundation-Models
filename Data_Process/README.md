## How to Use:

1. Ensure all datasets are correctly set up and paths are configured in the scripts.
2. Run the desired script(s) based on the task:
   - For Data split to train and validate for scFoundation version: `.py`
   - For few-shot for scFoundation version: `scGPT_run_all_celltypeannot_fewshot.py`
   - For ablation studies: `scGPT_run_all_celltypeannot_nopretrain{_freeze}.py`
3. Use `create_figures_and_tables.ipynb` to generate visualizations and tables from the results.
4. Use `create_figures_and_tables.ipynb` to generate visualizations and tables from the results.

For further details, refer to the comments within each script or the associated [documentation](#).



## Dataset Overview

| **Dataset**   | **Train** | **Test** |
|----------------|-----------|----------|
| **Lung-Kim**   | 23,185    | 7,387    |
| **Covid-19**   | 15,997    | 4,003    |
| **Myeloid**    | 9,748     | 3,430    |
| **hPancreas**  | 10,600    | 4,218    |
| **MS**         | 7,844     | 13,468   |
