import torch
import datasets
from datasets import load_dataset
mnist = load_dataset('mnist')

mnist.save_to_disk("datasets")
