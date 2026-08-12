# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor round-trip coverage for the Omni connector serializer."""

import pytest
import torch

from vllm_omni.distributed.omni_connectors.utils.serialization import (
    OmniMsgpackDecoder,
    OmniMsgpackEncoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "value",
    [
        torch.arange(24, dtype=torch.float32).reshape(4, 6),
        torch.arange(24, dtype=torch.float32).reshape(4, 6).transpose(0, 1),
        torch.empty((0, 8), dtype=torch.float32),
    ],
    ids=["contiguous-2d", "non-contiguous-2d", "empty-2d"],
)
def test_fp32_2d_tensor_round_trip_preserves_contract(value: torch.Tensor) -> None:
    encoded = OmniMsgpackEncoder().encode(value)
    decoded = OmniMsgpackDecoder().decode(encoded)

    assert isinstance(decoded, torch.Tensor)
    assert decoded.dtype == value.dtype
    assert decoded.shape == value.shape
    assert torch.equal(decoded, value)
