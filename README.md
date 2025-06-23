# Machine Learning Institute - Week 3 - Read digits

This week, we are using the MNIST dataset to train something which can read multiple digits from an image, using transformer based architecture/s.

# Set-up

* Install the [git lfs](https://git-lfs.com/) extension **before cloning this repository**
* Install the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)

Then install dependencies with:

```bash
uv sync --all-packages --dev
```

# Model Training

Run the following, with an optional `--model "model_name"` parameter

```bash
uv run -m model.start_train
```