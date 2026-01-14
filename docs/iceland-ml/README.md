# Machine Learning for Earth Observation powered by Supercomputers

## Course: TÖV606M

Welcome to the course! This module guides you through building a complete ML pipeline for Earth Observation using Sentinel-2 imagery, land cover classification, and HPC infrastructure at Jülich Supercomputing Centre.

### Course Details
- **Credits:** 6 ECTS
- **Instructors:** 
  - Gabriele Cavallaro (gcavallaro@hi.is) - Course Lead
  - Rocco Sedona (r.sedona@fz-juelich.de) - Technical Lead
  - Samy Hashim (s.hashim@fz-juelich.de) - Lab Instructor
  - Ehsan Zandi (e.zandi@fz-juelich.de)
- **Semester:** Spring 2025-2026
- **Modality:** Mixed in-person/online (Iceland + Germany)
- **HPC Resources:** JURECA (JSC/Judoor account required)

### Learning Outcomes
By completing this course, you will be able to:
- Access and manage HPC resources (Judoor, JURECA, SLURM)
- Acquire and preprocess satellite imagery (Sentinel-2 via Google Earth Engine)
- Build and train deep learning models for land cover classification
- Evaluate model performance using industry-standard metrics
- Deploy ML workflows on supercomputer infrastructure
- Fine-tune geospatial foundation models (TerraTorch, Prithvi)
- Apply models to generate classification maps and visualizations

---

## Lab Structure (8 Sessions × 120 minutes)

Each lab builds progressively toward a complete ML pipeline.

### Lab 1: Judoor Account and Access to HPC
**Week 1 | Duration: 120 min | Mode: Online**

**Topics:**
- Introduction to Jülich Supercomputing Centre (JSC)
- Creating and activating Judoor accounts
- SSH key setup and authentication
- First login to JURECA supercomputer
- Understanding HPC filesystem (home, project, scratch)
- Basic SLURM commands

**Deliverables:**
- ✅ Active Judoor account
- ✅ Membership in `training2600` project
- ✅ Successful SSH connection to JURECA
- ✅ Personal workspace directory structure

📓 **Notebook:** [`lab1_judoor_hpc_access.ipynb`](../../notebooks/iceland-ml/lab1_judoor_hpc_access.ipynb)

---

### Lab 2: Jupyter-JSC and Git Basics
**Week 3 | Duration: 120 min | Mode: Online**

**Topics:**
- Launching Jupyter-JSC (web-based JupyterLab on HPC)
- Git fundamentals for version control
- Cloning course repository
- Creating Python virtual environments
- Registering custom Jupyter kernels
- First notebook execution on HPC

**Deliverables:**
- ✅ Running Jupyter-JSC session
- ✅ Cloned course repository
- ✅ Custom Python kernel (ML-EO Course)
- ✅ Git identity configuration

📓 **Notebook:** [`lab2_jupyter_jsc_git.ipynb`](../../notebooks/iceland-ml/lab2_jupyter_jsc_git.ipynb)

---

### Lab 3: Google Earth Engine - Sentinel-2 Data Acquisition
**Week 6 | Duration: 120 min | Mode: Online**

**Topics:**
- Google Earth Engine (GEE) setup and authentication
- Defining Area of Interest (AOI) in Iceland
- Querying Sentinel-2 image collections
- Cloud filtering and scene selection
- Visualizing satellite imagery (RGB, false color)
- Exporting imagery for ML pipeline

**Deliverables:**
- ✅ GEE authenticated account
- ✅ 4 Sentinel-2 scenes (summer 2024, <20% cloud cover)
- ✅ Scene metadata (dates, cloud cover, bands)
- ✅ Downloaded GeoTIFF imagery

📓 **Notebook:** [`lab3_gee_sentinel2_acquisition.ipynb`](../../notebooks/iceland-ml/lab3_gee_sentinel2_acquisition.ipynb)

---

### Lab 4: Data Preprocessing
**Week 8 | Duration: 120 min | Mode: Online**

**Topics:**
- Loading and inspecting GeoTIFF imagery
- Normalization techniques (min-max, standardization, percentile clipping)
- Matching imagery with CORINE land cover labels
- Creating train/validation/test splits (70/15/15)
- Saving intermediate artifacts for downstream use

**Deliverables:**
- ✅ Normalized imagery ready for patching
- ✅ Train/val/test split definitions
- ✅ Normalization parameters (saved for inference)
- ✅ Dataset metadata (bands, classes, dimensions)

📓 **Notebook:** [`lab4_preprocessing_patches.ipynb`](../../notebooks/iceland-ml/lab4_preprocessing_patches.ipynb) — preprocessing section

---

### Lab 5: Patch Extraction
**Week 9 | Duration: 120 min | Mode: Online**

**Topics:**
- Extracting fixed-size patches (224×224) for deep learning
- Writing clean patch metadata
- Balancing classes across train/val/test splits
- Persisting patches efficiently (NumPy/NPZ)

**Deliverables:**
- ✅ Patch datasets for train/val/test
- ✅ Patch-level metadata and integrity checks
- ✅ Saved patch arrays ready for training

📓 **Notebook:** [`lab4_preprocessing_patches.ipynb`](../../notebooks/iceland-ml/lab4_preprocessing_patches.ipynb) — patch extraction section

---

### Lab 6: Baseline Model Training
**Week 11 | Duration: 120 min | Mode: Online**

**Topics:**
- Building CNN classifier with PyTorch
- Creating custom PyTorch Datasets and DataLoaders
- Training loop implementation (forward/backward passes)
- Learning rate scheduling and early stopping
- Model checkpointing (save best model)
- Submitting GPU training jobs via SLURM

**Deliverables:**
- ✅ Trained baseline CNN model
- ✅ Training curves (loss, accuracy)
- ✅ Best model checkpoint
- ✅ SLURM submission script for batch jobs

📓 **Notebook:** [`lab5.1_baseline_training.ipynb`](../../notebooks/iceland-ml/lab5.1_baseline_training.ipynb)

---

### Lab 7: Model Evaluation Metrics
**Week 12 | Duration: 120 min | Mode: Online**

**Topics:**
- Loading trained models and test data
- Generating predictions on test set
- Calculating classification metrics (accuracy, precision, recall, F1)
- Confusion matrix analysis
- Visualizing correct and incorrect predictions
- Per-class performance analysis
- Identifying model strengths and weaknesses

**Deliverables:**
- ✅ Comprehensive evaluation report
- ✅ Confusion matrix visualizations
- ✅ Per-class performance metrics
- ✅ Sample predictions (correct/incorrect)
- ✅ Recommendations for improvement

📓 **Notebook:** [`lab5.2_model_evaluation.ipynb`](../../notebooks/iceland-ml/lab5.2_model_evaluation.ipynb)

---

### Lab 8: TerraTorch Fine-tuning
**Week 13 | Duration: 120 min | Mode: Online**

**Topics:**
- Overview of TerraTorch and geospatial foundation models
- Preparing TerraTorch config for Sentinel-2 land cover
- Running fine-tuning on JURECA (CLI + SLURM submission)
- Logging, checkpoints, and experiment tracking

**Deliverables:**
- ✅ TerraTorch config tailored to the course dataset
- ✅ Fine-tuned model checkpoint
- ✅ Notes on hyperparameters and runtime performance

📓 **Notebook:** [`lab6_finetune.ipynb`](../../notebooks/iceland-ml/lab6_finetune.ipynb)

---

## Getting Started

### Required Knowledge
- Python programming (intermediate level)
- Basic machine learning concepts
- Familiarity with NumPy, Matplotlib
- Linux command line basics

### Required Accounts
- **Judoor Account:** https://judoor.fz-juelich.de/register
- **Google Earth Engine:** https://earthengine.google.com/signup
- **GitHub Account:** For cloning course repository

### Software Requirements
- SSH client (Terminal on Linux/Mac, PuTTY on Windows)
- Modern web browser (for Jupyter-JSC)
- Git (for version control)

---

## Resources

### Notebooks
All lab notebooks are available in [`notebooks/iceland-ml/`](../../notebooks/iceland-ml/):
- `lab1_judoor_hpc_access.ipynb`
- `lab2_jupyter_jsc_git.ipynb`
- `lab3_gee_sentinel2_acquisition.ipynb`
- `lab4.1_preprocessing.ipynb`
- `lab4.2_preprocessing_patches.ipynb`
- `lab5.1_baseline_training.ipynb`
- `lab5.2_model_evaluation.ipynb`
- `lab6_finetune.ipynb`

### Documentation
- **JSC Documentation:** https://apps.fz-juelich.de/jsc/hps/jureca/
- **Judoor Portal:** https://judoor.fz-juelich.de
- **Google Earth Engine:** https://developers.google.com/earth-engine
- **PyTorch Tutorials:** https://pytorch.org/tutorials/

### Communication
- **Slack Channel:** [Invite link provided by instructors]
- **Email Support:** s.hashim@fz-juelich.de
- **Office Hours:** By appointment (3 days advance notice)

---

## Assessment

### Lab Participation
- Complete all 8 lab exercises
- Submit working code and results
- Document preprocessing and training choices

### Final Project
- Apply learned techniques to custom AOI
- Train and evaluate classification model
- Present results (Week 14)

---

## Tips for Success

1. **Start Early:** HPC account setup takes time
2. **Test Incrementally:** Run code step-by-step, don't wait until deadline
3. **Ask Questions:** Use Slack channel for quick help
4. **Save Often:** Use checkpoints, Git commits, and backups
5. **Monitor Resources:** Check SLURM job status, GPU utilization
6. **Document Work:** Keep notes on experiments and results

---

## Next Steps

1. **Before Lab 1:** Create Judoor account (can take 1-2 days for approval)
2. **Before Lab 3:** Sign up for Google Earth Engine
3. **During Labs:** Follow notebooks sequentially, complete exercises
4. **After Labs:** Experiment with different datasets, architectures, hyperparameters

---

**Questions?** Contact Samy Hashim at s.hashim@fz-juelich.de

**Good luck, and enjoy your journey into ML for Earth Observation!** 🚀🛰️🌍