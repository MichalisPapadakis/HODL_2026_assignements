# Maze Path Prediction with Graph Neural Networks

This project trains a Graph Neural Network (GNN) to predict shortest-path nodes in randomly generated maze trees.

Course context:
- Deep Learning Course ETH
- Graph Neural Networks

## Project Overview

- Graphs are generated from random spanning trees on grid graphs.
- Node features are one-hot source/target indicators.
- Labels mark whether each node belongs to the shortest path between source and target.
- The core model (`MazeGNN`) uses message passing with input-memory fusion to preserve the original encoded input across convolution steps.

## Repository Files

- `challenge4.py`: model, dataset generation, training, evaluation, and plotting.
- `challenge4_cv.py`: parallel grid search over `hidden_dim`, `dropout`, and aggregation mode.
- `generated_paths/`: qualitative prediction plots by grid size.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and Evaluate

Run standard training and size-bucket evaluation:

```bash
python challenge4.py
```

This trains on `4x4` graphs and evaluates on `4x4`, `8x8`, `16x16`, and `32x32`.

## Hyperparameter Search

Run parallel grid search (CPU workers) with F1-based model selection:

```bash
python challenge4_cv.py
```

The search:
- trains one model per configuration on `4x4` data,
- evaluates node F1 on `4x4`, `8x8`, `16x16`, `32x32`,
- selects the best configuration by mean F1 across sizes.

## Indicative Solutions

These are example configurations that can be used as starting points:

- `hidden_dim=64`, `dropout=0.2`, `aggr="add"`
- `hidden_dim=32`, `dropout=0.2`, `aggr="add"`
- `hidden_dim=32`, `dropout=0.3`, `aggr="add"`

Actual scores vary with random seed and training budget.

## Notes

- Exact full-graph correctness is a strict metric; node-level F1 is often a better signal of learning quality across larger grids.
- The grid-search script keeps workers on CPU to avoid CUDA multiprocessing conflicts.
