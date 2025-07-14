# 3D Masked Autoencoders for Volumetric Medical Imaging Data

This repository provides a **3D extension of the Masked Autoencoder (MAE) framework**, designed for self-supervised pretraining on **volumetric medical imaging data** (e.g., CT scans). Our method extends MAE to 3D by incorporating **custom volumetric patch embedding** and **Transformer-based feature learning**, enabling efficient representation learning for medical imaging applications.

As part of this framework, we introduce **TANGERINE** (*Thoracic Autoencoder Network Generating Embeddings for Radiological Interpretation for Numerous End-tasks*), a **ViT-Large model pretrained on 98,000 chest CT volumes** for lung screening. TANGERINE demonstrates the effectiveness of this framework and is described in detail in our paper (citation below). We provide the **pretrained encoder and decoder weights**, which can be used to initialize fine-tuning for a variety of downstream tasks.


## Key Features

- **3D Extension of MAE**  
  - Adapts the MAE framework for 3D volumetric data.  
  - Employs a specialized **3D patch embedding module** for improved spatial feature extraction.  

- **Computationally Efficient Pretraining**  
  - Utilizes **high masking ratios** to reduce training memory consumption.  
  - Enables **scalable training on large-scale 3D datasets**.  

- **Pretrained ViT Large Model**  
  - TANGERINE, our pretrained ViT-Large model, was trained on 98,000 chest CT volumes for thoracic screening, as detailed in our paper.  
  - Pretrained TANGERINE **encoder and decoder weights** are available at: [Full access URL LINK coming soon](https://drive.google.com/drive/folders/1hESpODUMGY5572jDuZBB2QHiOf0ac5tO?usp=share_link)  
  - This pretrained model can be **readily finetuned** for a wide range of **downstream tasks**.

- **Flexible Finetuning and Inference**  
  - Includes scripts for **supervised finetuning** on downstream classification and segmentation tasks.  
  - Supports **efficient inference** using learned volumetric representations.  

---

## Installation

To use this repository, install the necessary dependencies and set up the environment.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/niccolo246/3D-MAE-MedImaging
   cd 3D-MAE-MedImaging
   ```

2. **Install dependencies:**  
   Ensure you have **Python 3.7+** and a compatible **PyTorch version** installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

For additional dependency requirements, refer to `requirements.txt`.

---

## Pretraining

Pretraining is performed using **distributed training** across multiple GPUs for efficiency. The following script launches pretraining with `torchrun` (4 GPUs):

```bash
# Check GPU availability
nvidia-smi

# Optional: Disable NCCL P2P if needed
export NCCL_P2P_DISABLE=1

# Assign a free port for distributed training
find_free_port() {
    while true; do
        PORT=$(shuf -i 20000-65000 -n 1)
        ss -lpn | grep ":$PORT " > /dev/null
        if [ $? -ne 0 ]; then
            echo $PORT
            return 0
        fi
    done
}
MASTER_PORT=$(find_free_port)
echo "Using port: $MASTER_PORT"

# Run pretraining
torchrun --standalone --nproc_per_node=4 --nnodes=1 --master_port=$MASTER_PORT path/to/main_pretrain.py
```

Modify `path/to/main_pretrain.py` based on your dataset and training parameters.

---

## Finetuning

To adapt the pre-trained model for downstream tasks, finetuning is performed as follows:

```bash
torchrun --standalone --nproc_per_node=4 --nnodes=1 --master_port=$MASTER_PORT path/to/main_finetune.py \
    --finetune /path/to/pretrained_checkpoint.pth \
    --additional_finetune_args
```

Modify `--additional_finetune_args` based on task-specific requirements.

---

## Inference and Prediction

For model inference on new volumetric datasets:

```bash
python3 main_predict.py \
    --input_csv /path/to/input.csv \
    --output_csv /path/to/output_predictions.csv \
    --finetune /path/to/pretrained_checkpoint.pth
```

This script loads the finetuned model and generates predictions.

---

## Custom Dataset Handling

**Important:** Users must create a **custom dataset class** (`Custom3DDataset`) depending on their **data structure**.  
- The dataset class for **pretraining** should be defined in:  
  **`datasets_three_d.py`**  
- The dataset class for **finetuning** should be defined in:  
  **`datasets_three_d_fine.py`**

Each user should modify `Custom3DDataset` to correctly **load, preprocess, and format their data** based on their dataset structure.

---

## Technical Details

### **3D Data Handling**  
- The dataset loader utilizes **SimpleITK** for reading **NIfTI** medical images.  
- Ensures correct axis ordering **([Depth, Height, Width])** for volumetric representation.  
- Includes **optional resampling functions** to standardize input dimensions.

#### **Resampling to 256x256x256**  
An **example resampling function** is provided in `datasets_three_d_fine.py` to **resize input volumes to 256×256×256 resolution**.  
Modify this function as needed to fit specific dataset characteristics.

### **Training and Sampling Strategy**  
- For **single-GPU training**, `DataLoader` is configured with `shuffle=True`.  
- In **distributed training**, `DistributedSampler` is recommended to partition data across GPUs.  
- *(Note: The `DistributedSampler` is included but commented out for single-GPU training.)*

### **Model Architecture**
- Utilizes **Transformer-based MAE architecture** for volumetric feature extraction.  
- Implements **custom 3D patch embedding** to handle medical imaging modalities.  
- Incorporates **high masking ratios** to enhance self-supervised learning efficiency.

---

## Pretrained Model Weights

We provide **TANGERINE pretrained ViT-Large weights** for both the **encoder** and **decoder**, available at the following link:

Coming soon...
[Full access URL LINK coming soon](https://drive.google.com/drive/folders/1hESpODUMGY5572jDuZBB2QHiOf0ac5tO?usp=share_link)


These weights can be directly used for **finetuning** across a wide range of downstream tasks, including **classification**, **segmentation**, and **detection**.

### Example usage

```bash
torchrun --standalone --nproc_per_node=4 --nnodes=1 --master_port=$MASTER_PORT path/to/main_finetune.py \
    --finetune path/to/pretrained_checkpoint.pth \
    --additional_finetune_args
```

---

## Citation & License

This project is licensed under the **CC-BY-NC 4.0** license.  

If you use this repository in academic work, please cite:

```
Pending
```

---

## Contact

For questions or contributions, please contact **niccolo.mcconnell.17@ucl.ac.uk** or open an issue on GitHub.


