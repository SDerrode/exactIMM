"""
prg/filter
==========
Fast optimal filter for Gaussian Switching Systems (Option B), the exact
O(N K²) fixed-interval smoothers on the exactness domains, and the
observed-chain likelihood / between-family LRT built on the same kernels.
"""

from prg.filter.gss_filter import FilterResult, GSSFilter
from prg.filter.gss_smoother import (
    chain_log_likelihood,
    constant_gain_smoother,
    family_lrt,
    lag1_constant_gain_smoother,
    regime_smoother,
    reweight_smoother,
)

__all__ = [
    "GSSFilter",
    "FilterResult",
    "regime_smoother",
    "reweight_smoother",
    "constant_gain_smoother",
    "lag1_constant_gain_smoother",
    "chain_log_likelihood",
    "family_lrt",
]
