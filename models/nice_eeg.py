# Modified from https://github.com/eeyhsong/NICE-EEG

# Builtin imports
from typing import Literal


# External imports
import torch
from torch import nn, Tensor
from einops.layers.torch import Rearrange
from einops import rearrange
from torch_geometric.nn import GATConv
from torch.nn import functional as F
from pydantic import BaseModel
from huggingface_hub import PyTorchModelHubMixin
from accelerate import Accelerator
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


"""
The implementation of spatial modules, self-attention (SA) and graph attention (GA).
It's a consice and simple use of channel-wise attention / graph.

! please install torch_geometric and insert the following code in nice_stand.py
"""


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = self.tsconv(x)
        x = self.projection(x)
        return x


# Revised for the smaller dimensions of my model
class SmallPatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, num_channels=32):
        super().__init__()

        # After conv (1,25): 78 - 25 + 1 = 54 timesteps preserved

        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),  # Temporal conv
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (num_channels, 1), (1, 1)),  # Spatial conv
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


# class FlattenHead(nn.Module):
#     def __init__(self, emb_size: int, num_classes: int):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         # The patch embedding outputs (1, emb_size) spatial dims, so flatten gives emb_size
#         self.fc = nn.Linear(emb_size, num_classes)

#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.fc(x)
#         return x


class channel_attention(nn.Module):
    def __init__(
        self,
        sequence_num,
        num_channels,
        dropout_rate,
        inter=30,
    ):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(
            self.sequence_num / self.inter
        )  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.LayerNorm(num_channels),
            nn.Dropout(dropout_rate),
        )
        self.key = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.LayerNorm(num_channels),
            nn.Dropout(dropout_rate),
        )

        self.projection = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.LayerNorm(num_channels),
            nn.Dropout(dropout_rate),
        )

        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, "b o c s->b o s c")
        temp_query = rearrange(self.query(temp), "b o s c -> b o c s")
        temp_key = rearrange(self.key(temp), "b o s c -> b o c s")

        channel_query = temp_query
        channel_key = temp_key

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = (
            torch.einsum("b o c s, b o m s -> b o c m", channel_query, channel_key)
            / scaling
        )

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum("b o c s, b o c m -> b o c s", x, channel_atten_score)

        out = rearrange(out, "b o c s -> b o s c")
        out = self.projection(out)
        out = rearrange(out, "b o s c -> b o c s")
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum(
            "bhqd, bhkd -> bhqk", queries, keys
        )  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size,
        num_heads=10,
        drop_p=0.5,
        forward_expansion=4,
        forward_drop_p=0.5,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class EEG_GAT(nn.Module):
    def __init__(self, num_channels: int, in_channels=250, out_channels=250):
        super(EEG_GAT, self).__init__()
        self.num_channels = num_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GATConv(
            in_channels=in_channels, out_channels=out_channels, heads=1
        )
        # self.conv2 = GATConv(in_channels=out_channels, out_channels=out_channels, heads=1)

        # Create a list of tuples representing all possible edges between channels
        self.edge_index_list = torch.Tensor(
            [
                (i, j)
                for i in range(self.num_channels)
                for j in range(self.num_channels)
                if i != j
            ]
        ).cuda()
        # Convert the list of tuples to a tensor
        self.edge_index = (
            torch.tensor(self.edge_index_list, dtype=torch.long).t().contiguous().cuda()
        )

    def forward(self, x):
        batch_size, _, num_channels, num_features = x.size()
        x = x.view(batch_size * num_channels, num_features)
        x = self.conv1(x, self.edge_index)
        x = x.view(batch_size, num_channels, -1)
        x = x.unsqueeze(1)

        return x


class EncEEGConfig(BaseModel):
    spatial: Literal["ga"] | Literal["channel"]
    sequence_len: int
    num_classes: int
    num_channels: int
    emb_size: int = 40
    dropout_rate: float = 0.3


# TODO: try out their transformer version as well
class Enc_EEG(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        *,
        spatial: str,
        emb_size: int,
        num_classes: int,
        num_channels: int,
        dropout_rate: float,
        sequence_len: int,
    ):
        super().__init__()

        if spatial == "channel":
            spatial_module = nn.Sequential(
                nn.LayerNorm(sequence_len),
                channel_attention(
                    sequence_num=sequence_len,
                    num_channels=num_channels,
                    dropout_rate=dropout_rate,
                ),
            )
        elif spatial == "ga":
            spatial_module = EEG_GAT(
                # Confusingly, this calls both sequence and channels "channels "
                in_channels=sequence_len,
                out_channels=sequence_len,
                num_channels=num_channels,
            )
        else:
            raise ValueError(f"Bad spatial={spatial}")

        self.spatial_block = ResidualAdd(
            nn.Sequential(
                spatial_module,
                nn.Dropout(dropout_rate),
            )
        )
        self.emb = SmallPatchEmbedding(emb_size, num_channels=num_channels)
        self.flatten = nn.Flatten()

        # TODO: I don't quite understnad where this comes from, the emb is part of it
        self.proj = nn.Linear(in_features=127 * emb_size, out_features=num_classes)

    def forward(self, x):
        # Add channel dimension if needed: (B, C, T) -> (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.spatial_block(x)
        x = self.emb(x)
        # print(f"Emb shape {x.shape}")
        x = self.flatten(x)
        # print(f"Post flatten shape {x.shape}")
        x = self.proj(x)

        return x

    @classmethod
    def from_config(cls, config: EncEEGConfig):
        return cls(**config.model_dump())


class EncEEGTrainer:
    def __init__(
        self,
        *,
        classifier: Enc_EEG,
        accelerator: Accelerator,
        scheduler: LRScheduler,
        optimizer: Optimizer,
    ):
        self.classifier = classifier
        self.accelerator = accelerator
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()

    def step(self, x, y):
        self.optimizer.zero_grad()

        logits = self.classifier(x)
        loss = self.criterion(logits, y)

        self.accelerator.backward(loss)

        self.optimizer.step()
        self.scheduler.step()

        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        num_correct = (predictions == y).sum().item()
        total = y.shape[0]
        accuracy = num_correct / total

        self.accelerator.log(
            {"loss": loss, "lr": self.scheduler.get_last_lr()[0], "accuracy": accuracy}
        )

        return {
            "loss": loss,
            "num_correct": num_correct,
            "total": total,
            "accuracy": accuracy,
        }
