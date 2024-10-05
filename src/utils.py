import torch

def save_model(model, path):
    """
    Saves the trained model to the specified file path.
    :param model: Trained model
    :param path: File path to save the model
    """
    torch.save(model.state_dict(), path)

def load_model(path, model_class):
    """
    Loads a pre-trained model from the specified file path.
    :param path: File path to load the model from
    :param model_class: Class of the model (e.g., GEARSage, TGAT)
    :return: Loaded model
    """
    model = model_class()
    model.load_state_dict(torch.load(path))
    return model
