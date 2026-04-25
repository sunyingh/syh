from __future__ import absolute_import, division, print_function

from options import MonodepthOptions as BaseMonodepthOptions
from options import str2bool


class MonodepthOptions(BaseMonodepthOptions):
    """Ubuntu training options extended with PCC flags."""

    def __init__(self):
        super().__init__()

        self.parser.add_argument(
            "--use_pcc",
            type=str2bool,
            help="enable photometric-invariant cycle constraint",
            default=True,
        )
        self.parser.add_argument(
            "--pcc_weight",
            type=float,
            help="weight for PCC appearance term",
            default=0.01,
        )
        self.parser.add_argument(
            "--pcc_use_cycle",
            type=str2bool,
            help="enable cycle-consistency term in PCC",
            default=True,
        )
        self.parser.add_argument(
            "--pcc_cycle_weight",
            type=float,
            help="relative weight for PCC cycle term",
            default=0.5,
        )
        self.parser.add_argument(
            "--pcc_window",
            type=int,
            help="local normalization and NCC window size for PCC",
            default=5,
        )
