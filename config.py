from os import path
import argparse

module_path = path.dirname(path.abspath(__file__))

from utils.dataset import load_textual_sp
from utils.dataset_preprocess import *


load_dataset = {

    'sp': load_textual_sp
}


preprocess_original_dataset = {

    'sp': preprocess_function_original_sp
}


preprocess_train_dataset = {

    'sp': preprocess_function_generator_sp
}

preprocess_test_dataset = {

    'sp': preprocess_test_function_generator_sp
}

instruction_len = {
    'sp': 450,
    'sc': 45,
    'bgm': 35,
    'mts': 50,
}

original_len = {
    'sp': 450,
    'sc': 60,
    'bgm': 60,
    'mts': 90,
}

task_level = {
    'sp': 'node',
    'sc': 'node',
    'bgm': 'graph',
    'mts': 'node',
}


def parse_args_llama():
    parser = argparse.ArgumentParser(description="GraphLLM")
    parser.add_argument("--prediction_type", type=str, default="next_node",help='next_node or current_node')
    parser.add_argument("--project", type=str, default="project_GraphLLM")
    parser.add_argument("--exp_num", type=int, default=1)
    parser.add_argument("--model_name", type=str, default='Meta-Llama-3.1-8B')

    parser.add_argument("--dataset", type=str, default='mol')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.1)


    parser.add_argument("--adapter_len", type=int, default=5)
    parser.add_argument("--adapter_dim", type=int, default=768)
    parser.add_argument("--adapter_n_heads", type=int, default=6)


    parser.add_argument("--n_decoder_layers", type=int, default=4)
    parser.add_argument("--n_encoder_layers", type=int, default=4)
    parser.add_argument("--n_mp_layers", type=int, default=4)


    # Model Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_steps", type=int, default=2)


    # Learning Rate Scheduler
    parser.add_argument("--num_epochs", type=int, default=15)


    parser.add_argument("--warmup_epochs", type=float, default=5)

    # RRWP
    parser.add_argument("--rrwp", type=int, default=8)

    # Inference
    parser.add_argument("--eval_batch_size", type=int, default=80)

    args = parser.parse_args()
    return args



