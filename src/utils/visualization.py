import matplotlib.pyplot as plt
import torch


def visualize_sk_attention(attention_weights):
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.mean(dim=0).squeeze().cpu().numpy()

    plt.bar(range(len(attention_weights)), attention_weights)
    plt.xlabel("Branch index")
    plt.ylabel("Attention weight")
    plt.title("Selective Kernel Attention")
    plt.show()
