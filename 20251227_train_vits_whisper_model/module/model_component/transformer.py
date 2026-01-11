import math
from typing import Any, Optional
import torch
from torch import nn
from torch.nn import functional as F

from utils.model import convert_pad_shape
from module.model_component.normalization import LayerNorm