from dataclasses import dataclass

@dataclass
class SKConfig:
    M = 2
    G = 32
    r = 16

    kernel_sizes = (3, 5)
    dilations = (1, 2)

    in_channels = 64
    num_classes = 1000
