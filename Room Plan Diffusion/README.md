# Room Plan Diffusion: Generating Indoor Furniture Layouts

**National Science and Technology Council (NSTC) Project**
Project ID: **NSTC-114-2218-E-A49-026/**

This repository contains the official implementation of **Room Plan Diffusion**, a diffusion-based framework for generating structured **indoor room layouts with furniture arrangements**. The project is part of the NSTC research program and aims to advance **Generative AI for 3D indoor scene synthesis**, enabling applications in **room planning, furniture layout generation, and text-conditioned scene design**.

---

## 🏠 Introduction

**Room Plan Diffusion: Generating Indoor Furniture Layouts** is a diffusion-based generative framework dedicated to **indoor scene synthesis and furniture arrangement**. This project is part of the **National Science and Technology Council (NSTC) research program (NSTC-114-2218-E-A49-026/)**. By leveraging large-scale 3D datasets such as **3D-FRONT** and **3D-FUTURE**, along with **semantic-guided generation techniques**, the system is able to synthesize structured indoor layouts from either random initialization or natural language prompts.

We have successfully developed a **Text-Guided Denoising Pipeline**, which achieves **96% accuracy across 50 test cases**. This demonstrates the ability of Room Plan Diffusion to effectively align **textual descriptions with 3D spatial layouts**, producing realistic and semantically consistent room plans.

Key features of the project include:

* **Unconditional Scene Generation**: Generate complete room layouts from random seeds.
* **Text-Conditioned Generation**: Synthesize indoor scenes guided by natural language descriptions.
* **Shape Autoencoder & Latent Codes**: Learn compact shape representations for furniture modeling and integration.
* **Evaluation Metrics**: Comprehensive assessment using FID, KID, IoU, symmetry measures, and precision/recall.

The system paves the way for applications in **interior design automation, furniture layout optimization, AR/VR scene creation, and robotic simulation environments**, providing a foundation for cross-domain research at the intersection of **Generative AI and 3D scene modeling**.

## 🔬 Latest Progress

We have successfully implemented the **Text-Guided Denoising Pipeline** for **Room Plan Diffusion**, enabling controllable scene synthesis from natural language descriptions.

* **Pipeline:** Text-guided denoising for indoor room and furniture layout generation
* **Evaluation:** On 50 test cases, the pipeline achieved **96% accuracy** in generating room plans that matched the target scene specifications
* **Implication:** This result demonstrates that Room Plan Diffusion can effectively align **textual prompts with 3D spatial layouts**, opening the door for practical applications in **interior design automation, AR/VR scene creation, and robotics simulation environments**

Pre-trained models and example prompts will be released soon to reproduce the text-conditioned results.

## 📦 Installation & Dependencies

Create a conda environment:

```bash
conda env create -f environment.yaml
conda activate RoomPlanDiffusion
```

Compile extension modules:

```bash
python setup.py build_ext --inplace
pip install -e .
```

Install **ChamferDistancePytorch**:

```bash
cd ChamferDistancePytorch/chamfer3D
python setup.py install
```

---

## 📂 Dataset

We train and evaluate our model using:

* [**3D-FRONT**](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset)
* [**3D-FUTURE**](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset)

Please follow the official instructions to download the datasets.

### 🔧 Preprocessing

**1. Pickle the 3D-FUTURE dataset:**

```bash
python pickle_threed_future_dataset.py path_to_output_dir path_to_3d_front_dataset_dir path_to_3d_future_dataset_dir path_to_3d_future_model_info --dataset_filtering room_type
```

**2. Pickle CAD point clouds:**

```bash
python pickle_threed_future_pointcloud.py path_to_output_dir path_to_3d_front_dataset_dir path_to_3d_future_dataset_dir path_to_3d_future_model_info --dataset_filtering room_type
```

**3. Train shape autoencoder:**

```bash
cd ./scripts
PATH_TO_SCENES=".../threed_front.pkl" python train_objautoencoder.py ../config/obj_autoencoder/bed_living_diningrooms_lat32.yaml output_dir --experiment_tag "bed_living_diningrooms_lat32" --with_wandb_logger
```

**4. Extract latent shape codes:**

```bash
PATH_TO_SCENES=".../threed_front.pkl" python generate_objautoencoder.py ../config/objautoencoder/bedrooms.yaml output_dir --experiment_tag "bed_living_diningrooms_lat32"
```

**5. Preprocess with latent shape codes:**

```bash
PATH_TO_SCENES=".../threed_front.pkl" python preprocess_data.py output_dir /path/to/3D-FRONT /path/to/3D-FUTURE-model /path/to/3D-FUTURE-model/model_info.json --dataset_filtering threed_front_livingroom --annotation_file ../config/livingroom_threed_front_splits.csv --add_objfeats
```

Note: Execute preprocessing separately for **bedrooms** and **living/dining rooms**.

---

## 🚀 Training & Generation

**Train Room Plan Diffusion:**

```bash
./run/train.sh
./run/train_text.sh
```

**Generate indoor scenes:**

```bash
./run/generate.sh
./run/generate_text.sh
```

Options:

* Add `--compute_intersec` to compute bbox IoU and symmetry metrics.
* Text-conditioned models assume **2–3 sentence prompts**; for longer prompts, re-training is recommended.

---

## 📊 Evaluation

**FID & KID (from 2D rendered images):**

```bash
python compute_fid_scores.py $ground_truth_bedrooms $generated_bedrooms ../config/bedroom_threed_front_splits.csv
python compute_fid_scores.py $ground_truth_diningrooms $generated_diningrooms ../config/diningroom_threed_front_splits.csv
```

**Improved Precision & Recall:**

```bash
python improved_precision_recall.py $ground_truth_bedrooms $generated_bedrooms ../config/bedroom_threed_front_splits.csv
python improved_precision_recall.py $ground_truth_diningrooms $generated_diningrooms ../config/diningroom_threed_front_splits.csv
```

---

## 📑 Citation

If you use this code in your research, please cite the NSTC project:

```
@misc{RoomPlanDiffusion2025,
  title     = {Room Plan Diffusion: Generating Indoor Furniture Layouts},
  author    = {Yuan-Fu Yang and Jun-Teng Chen},
  year      = {2025},
  note      = {NSTC Project ID: NSTC-114-2218-E-A49-026/}
}
```


