import json
import os


def model_path(cfg):
    """Generate a unique file path for the model run based on the number of json file.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing model parameters and settings.

    Returns
    -------
    str
        The unique file path for the final model.
    """
    model_dir = "models/"
    model_name = cfg.model.name + "_" + cfg.dataset.name
    os.makedirs(model_dir, exist_ok=True)
    version_file = os.path.join(model_dir, "models_version.json")
    if os.path.exists(version_file):
        with open(version_file, "rb") as f:
            models_version = json.load(f)
    else:
        models_version = {model_name: 0}

    if model_name not in models_version:
        models_version[model_name] = 0

    models_version[model_name] += 1
    with open(version_file, "w") as f:
        json.dump(models_version, f, indent=4)

    file_name = f"{model_name}_{models_version[model_name]}"
    file_path = os.path.join(model_dir, file_name)
    return file_path
