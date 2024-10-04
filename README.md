# Sammon Mapping

Sammon mapping is a nonlinear dimensionality reduction technique that preserves the structure of data by minimizing the difference between distances in high-dimensional space and their corresponding distances in a lower-dimensional projection.

## Code Explanation

The code is implemented in a simple way. You can run it from the command line and modify parameters such as:

- `datasetName`: Name of the dataset (default: `blobs`)
- `iter`: Number of iterations (default: `20`)
- `mf`: Learning factor (default: `0.3`)
- `initialization`: Initialization method (`random` or `orthogonal`, default: `random`)
- `dimension`: Number of dimensions for projection (`2` or `3`, default: `2`)

### Running the Code

You can run the code from the terminal (if you are in the project folder) without any parameters like this:

```
python3 main.py
```

In this case, the code will run with the default parameters:
- `datasetName=blobs`
- `iter=20`
- `mf=0.3`
- `initialization=random`
- `dimension=2`

### Modifying Parameters

If you want to change the parameters, you can specify them after `main.py` in the following order:

```
python3 main.py <datasetName> <iter> <mf> <initialization> <dimension>
```
For example:
```
python3 main.py circles 50 0.2 orthogonal 2
```

This command will run Sammon mapping on the `circles` dataset for `50` iterations, with a learning factor of `0.1`, using `orthogonal` initialization, and reducing the data to `3` dimensions.

### Input and Output

- **Input**:
  - CSV file containing the data positions: `../<datasetName>.csv`
  - CSV file containing the data labels: `../<datasetName>_labels.csv`

- **Output**:
  - CSV file with the resulting points: `<datasetName>Result.csv`
  - Generated plots:
    - Sammon map of the reduced data
    - Error E over iterations

### 3D Example

The code supports 3D visualization. To try this feature, set `dimension=3` as a parameter when running.

### Requirements

Ensure you have the following Python libraries installed:

- `numpy`
- `matplotlib`

