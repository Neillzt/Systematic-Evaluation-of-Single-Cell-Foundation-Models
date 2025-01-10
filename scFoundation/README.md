We provide different methods to evaluate the Foundation model.

## Script Features

The provided script includes the following functionalities:

### 1. Fine-Tuning and Few-Shot Training
- Supports **basic fine-tuning training** and **few-shot training**.
- Allows **freezing** or **updating** pre-trained weights.
- Provides an **expandable classification layer** that can be customized as needed.

### 2. Training from Scratch
- Reads the model structure and **randomly initializes all parameters** for training from scratch.

### 3. Zero-Shot Training (Reference Mapping)
- Inputs the reference (training set) and generates **gene embeddings** along with their corresponding known cell types.
- Uses **cosine distance** to predict cell types based on the gene embeddings of the test set.
- Implements the **K-NN method** for voting, where the parameter `K` can be preset by the user.

---

### How to Use
- Adjust the parameters for **fine-tuning**, **training from scratch**, or **zero-shot training** in the script as needed.
- Refer to the script comments for detailed instructions on parameter configuration and execution.

For more details, check the provided script or documentation.
