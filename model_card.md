---
license: cc-by-nc-4.0
tags:
  - medical-imaging
  - 3d-transformer
  - masked-autoencoder
  - vit
  - chest-ct
  - self-supervised-learning
datasets:
  - custom-private
pipeline_tag: feature-extraction
library_name: pytorch
model-index:
  - name: TANGERINE (ViT-Large)
---

# Model Card: TANGERINE (ViT-Large for 3D Medical Imaging)

## Model Overview
TANGERINE is a 3D Vision Transformer (ViT-Large) model pretrained using a self-supervised Masked Autoencoder (MAE) strategy on a large-scale chest CT dataset (98,000 volumes). It is designed for effective transfer learning in volumetric medical imaging tasks.

## Pretraining Details
- **Architecture**: ViT-Large
- **Pretraining method**: Masked Autoencoding (MAE)
- **Input size**: 256×256×256 CT volumes
- **Pretraining data**: 98,000 thoracic CT scans (private dataset)

## Intended Use
- **Recommended for**: Transfer learning on 3D medical imaging tasks (e.g., classification, segmentation)
- **Example downstream tasks**:
  - Lung cancer classification
  - Airway disease detection
  - Organ segmentation

## Limitations
- Model was trained exclusively on thoracic CT; generalisation to other anatomies is not guaranteed.

## File Details

- `mae_pretrained.pth`: Contains both **encoder** and **decoder** weights from the pretrained MAE model.
  - The **encoder** weights are intended for **downstream tasks** such as classification, segmentation, and detection.
  - The **decoder** is included for completeness but is **not required** for finetuning or inference on downstream tasks.

## Citation
TBD – please cite our paper if you use this model.

## Contact
Questions or issues? Contact [niccolo.mcconnell.17@ucl.ac.uk](mailto:niccolo.mcconnell.17@ucl.ac.uk)
