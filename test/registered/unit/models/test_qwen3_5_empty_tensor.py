"""Unit tests for Qwen3.5 empty hidden state handling."""

import types
import unittest

import torch

from sglang.srt.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5GatedDeltaNet,
    Qwen3_5LinearDecoderLayer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _SentinelCalled(RuntimeError):
    pass


def _raise(name):
    raise _SentinelCalled(f"{name} was called for an empty hidden_states tensor")


def _empty_hidden_states():
    return torch.empty((0, 64), dtype=torch.float32)


class TestQwen3_5EmptyTensor(CustomTestCase):
    def test_gated_delta_net_skips_empty_hidden_states(self):
        layer = object.__new__(Qwen3_5GatedDeltaNet)
        layer._forward_input_proj = types.MethodType(
            lambda self, hidden_states: _raise("_forward_input_proj"),
            layer,
        )

        output = Qwen3_5GatedDeltaNet.forward(layer, _empty_hidden_states(), object())

        self.assertEqual(output.shape, torch.Size([0, 64]))

    def test_linear_decoder_layer_skips_empty_hidden_states(self):
        layer = object.__new__(Qwen3_5LinearDecoderLayer)
        layer.layer_communicator = types.SimpleNamespace(
            prepare_attn_and_capture_last_layer_outputs=lambda *args, **kwargs: _raise(
                "prepare_attn_and_capture_last_layer_outputs"
            )
        )

        hidden_states, residual = Qwen3_5LinearDecoderLayer.forward(
            layer, _empty_hidden_states(), None, forward_batch=object()
        )

        self.assertEqual(hidden_states.shape, torch.Size([0, 64]))
        self.assertEqual(residual.shape, torch.Size([0, 64]))

    def test_attention_decoder_layer_skips_empty_hidden_states(self):
        layer = object.__new__(Qwen3_5AttentionDecoderLayer)
        layer.layer_communicator = types.SimpleNamespace(
            prepare_attn_and_capture_last_layer_outputs=lambda *args, **kwargs: _raise(
                "prepare_attn_and_capture_last_layer_outputs"
            )
        )
        positions = torch.empty((0,), dtype=torch.long)

        hidden_states, residual = Qwen3_5AttentionDecoderLayer.forward(
            layer, positions, _empty_hidden_states(), None, object()
        )

        self.assertEqual(hidden_states.shape, torch.Size([0, 64]))
        self.assertEqual(residual.shape, torch.Size([0, 64]))


if __name__ == "__main__":
    unittest.main()
