# dataset-curation-pipeline


A scalable pipeline for transforming large, noisy image datasets into high-quality training datasets using modern vision embeddings and vector similarity search.

This system processes millions of images and removes duplicates and near-duplicates, and selects a **diverse subset of images** suitable for training machine learning models.

In addition to filtering and sampling datasets, the pipeline also supports **targeted image retrieval**. This allows users to retrieve images from the dataset that are **semantically similar to a candidate image or folder of images.** This is useful when training models that require targeted examples of a specific object, environment, or failure cases.

The pipeline leverages vision models like Meta AI's DINOv2, pgVector, and sampling algorithms used in machine learning infrastructure.

# Problem Context


Many machine learning systems rely on large image datasets collected from real-world environments. These datasets often contain:

- near-duplicate images
- poor quality frames
- redundant samples
- heavy class imbalance
- large volumes of unlabelled data

In my particular use case, our robotics camera system can capture millions of images, but only a small subset may be useful for training. 

Training on unfiltered datasets introduces several issues:

### 1. Duplicate images

Datasets often contain many identical or near-identical images captured in sequence. Training on these adds no new info, wastes compute, and **risks overfitting the model.**

### 2. Lack of Diversity

Even after deduplication, many datasets lack visual diversity, which can cause models to overfit. The unfiltered dataset may also be unbalanced, in the sense that there may be considerably more images of object/environment A, compared to B, which would cause the model to overfit.

### 3. Dataset Sca;e

It will take someone days/weeks to manually select the best dataset to train on when we reach hundreds of thousands of images. A scalable automated filtering pipeline that can remove duplicates, store embeddings for efficient similarity search, select a maximally diverse subset of images, and support cloud-scale storage and processing is nesscary.

# Proposed Solution

The pipeline converts a large unfiltered image dataset into a curated training dataset through the following stages:
1. Feature extraction through vector embeddings
2. Image and Metadata storage
3. Duplicate Filtering
4. Dataset Sampling
5. Image Retrieval
