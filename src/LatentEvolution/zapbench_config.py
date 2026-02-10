"""zapbench configuration: condition metadata, data splits, and training params."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# condition metadata
# ---------------------------------------------------------------------------

ConditionName = Literal[
    "gain", "dots", "flash", "taxis", "turning",
    "position", "open_loop", "rotation", "dark",
]


class Condition(BaseModel):
    """static condition metadata."""
    name: ConditionName
    offset: tuple[int, int]  # [start, end) along T dimension
    padding: int             # timesteps excluded at start/end

    model_config = ConfigDict(extra="forbid")


# constant list derived from zapbench CONDITION_OFFSETS
CONDITIONS: list[Condition] = [
    Condition(name="gain", offset=(0, 649), padding=1),
    Condition(name="dots", offset=(649, 2422), padding=1),
    Condition(name="flash", offset=(2422, 3078), padding=1),
    Condition(name="taxis", offset=(3078, 3735), padding=1),
    Condition(name="turning", offset=(3735, 5047), padding=1),
    Condition(name="position", offset=(5047, 5638), padding=1),
    Condition(name="open_loop", offset=(5638, 6623), padding=1),
    Condition(name="rotation", offset=(6623, 7279), padding=1),
    Condition(name="dark", offset=(7279, 7879), padding=1),
]


# ---------------------------------------------------------------------------
# data split config
# ---------------------------------------------------------------------------


class ConditionSplit(BaseModel):
    """split for one condition (references by name)."""
    condition_name: ConditionName
    train: tuple[int, int] | None = None  # [start, end) relative to padded start
    val: tuple[int, int] | None = None
    test: tuple[int, int] | None = None

    model_config = ConfigDict(extra="forbid")


def _default_condition_splits() -> list[ConditionSplit]:
    """default splits: 70% train, 10% val, 20% test; taxis is holdout."""
    return [
        # gain: padded_len=647, train=453, val=65, test=129
        ConditionSplit(condition_name="gain", train=(0, 453), val=(453, 518), test=(518, 647)),
        # dots: padded_len=1771, train=1240, val=177, test=354
        ConditionSplit(condition_name="dots", train=(0, 1240), val=(1240, 1417), test=(1417, 1771)),
        # flash: padded_len=654, train=458, val=65, test=131
        ConditionSplit(condition_name="flash", train=(0, 458), val=(458, 523), test=(523, 654)),
        # taxis (holdout): padded_len=655, test only
        ConditionSplit(condition_name="taxis", train=None, val=None, test=(0, 655)),
        # turning: padded_len=1310, train=917, val=131, test=262
        ConditionSplit(condition_name="turning", train=(0, 917), val=(917, 1048), test=(1048, 1310)),
        # position: padded_len=589, train=412, val=59, test=118
        ConditionSplit(condition_name="position", train=(0, 412), val=(412, 471), test=(471, 589)),
        # open_loop: padded_len=983, train=688, val=98, test=197
        ConditionSplit(condition_name="open_loop", train=(0, 688), val=(688, 786), test=(786, 983)),
        # rotation: padded_len=654, train=458, val=65, test=131
        ConditionSplit(condition_name="rotation", train=(0, 458), val=(458, 523), test=(523, 654)),
        # dark: padded_len=598, train=418, val=60, test=120
        ConditionSplit(condition_name="dark", train=(0, 418), val=(418, 478), test=(478, 598)),
    ]


class DataSplit(BaseModel):
    """zapbench data split across all conditions.

    default: 70% train, 10% val, 20% test; taxis is holdout (test only).
    """
    conditions: list[ConditionSplit] = Field(default_factory=_default_condition_splits)

    model_config = ConfigDict(extra="forbid")

    def get_ranges(
        self,
        split_type: Literal["train", "val", "test"],
    ) -> list[tuple[ConditionName, int, int]]:
        """get (condition_name, abs_start, abs_end) for each condition.

        args:
            split_type: which split to get ranges for

        returns:
            list of (condition_name, abs_start, abs_end) tuples
        """
        ranges = []
        for cs in self.conditions:
            range_rel = getattr(cs, split_type)
            if range_rel is None:
                continue
            cond = next(c for c in CONDITIONS if c.name == cs.condition_name)
            padded_start = cond.offset[0] + cond.padding
            abs_start = padded_start + range_rel[0]
            abs_end = padded_start + range_rel[1]
            ranges.append((cs.condition_name, abs_start, abs_end))
        return ranges


# ---------------------------------------------------------------------------
# training config
# ---------------------------------------------------------------------------


class DataConfig(BaseModel):
    """data paths and preprocessing params."""
    traces_path: str = "/groups/saalfeld/saalfeldlab/zapbench-release/volumes/20240930/traces"
    ephys_path: str = "/groups/saalfeld/home/kumarv4/repos/zapbench/ephys.zarr"
    bin_size_ms: float = 40.0

    model_config = ConfigDict(extra="forbid")


class TrainConfig(BaseModel):
    """training hyperparameters."""
    seed: int = 135717
    fitting_window: int = 100  # time steps to predict (4s at 25 Hz)
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-5
    batches_per_epoch: int = 0  # 0 = 1 full pass over data

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# model config
# ---------------------------------------------------------------------------


class EncoderDecoderConfig(BaseModel):
    """symmetric encoder/decoder config."""
    hidden_units: int = 128
    hidden_layers: int = 2
    activation: str = "ReLU"

    model_config = ConfigDict(extra="forbid")


class EvolverConfig(BaseModel):
    """evolver: latent (L) -> latent (L) with residual connection."""
    hidden_units: int = 128
    hidden_layers: int = 2
    zero_init: bool = True  # start as identity (z_{t+1} = z_t)
    activation: str = "Tanh"  # tanh for stability in rollouts

    model_config = ConfigDict(extra="forbid")


class ModelConfig(BaseModel):
    """full EED model configuration."""
    num_neurons: int  # N - number of neurons in dataset
    latent_dims: int = 128  # L - latent space dimension
    encoder_decoder: EncoderDecoderConfig = Field(default_factory=EncoderDecoderConfig)
    evolver: EvolverConfig = Field(default_factory=EvolverConfig)

    model_config = ConfigDict(extra="forbid")
