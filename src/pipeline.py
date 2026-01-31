import torch
from config import SKConfig
from models.sknet_model import SKNet


def run_forward():
    config = SKConfig()
    model = SKNet(config)
    model.eval()

    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)


if __name__ == "__main__":
    run_forward()
