import pickle
from config import module_path
import torch
from transformers import LlamaTokenizer
from torch.utils.data import TensorDataset
import datasets

def load_textual_sp(prediction_type):
    dataset = datasets.load_from_disk(f"{module_path}/dataset/custom/{prediction_type}")
    split = pickle.load(open(f"{module_path}/dataset/custom/split.pkl", 'rb'))
    edge_index = pickle.load(open(f"{module_path}/dataset/custom/edge_index.pkl", 'rb'))
    return dataset, split, edge_index








