from lightglue import LightGlue, SuperPoint, DISK, SIFT
from lightglue.utils import load_image, rbd
import torch

torch.set_grad_enabled(False)


if __name__ == "__main__":
    print("::: LightGlue pipeline\n")
