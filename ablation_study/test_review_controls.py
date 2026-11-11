"""Torch-only regression tests for reviewer-requested InfoGate controls."""

import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import infogate_modules
import infogate_modules_fixed


IMPLEMENTATIONS = (infogate_modules_fixed, infogate_modules)


def _tiny_args(**overrides):
    args = {
        'text_dim': 2,
        'acoustic_dim': 2,
        'visual_dim': 2,
        'unified_dim': 4,
        'ib_hidden_dim': 4,
        'bottleneck_dim': 2,
        'num_heads': 1,
        'num_infogate_layers': 1,
        'dropout_prob': 0.0,
        'use_l_lib': False,
        'use_l_rib': False,
    }
    args.update(overrides)
    return args


class ReviewControlTests(unittest.TestCase):
    def test_default_token_gumbel_matches_original_formulas(self):
        pooled = [
            torch.tensor([[0.2, -0.1], [0.7, 0.4], [-0.3, 0.8]]),
            torch.tensor([[0.5, 0.9], [-0.2, 0.6], [0.1, -0.4]]),
            torch.tensor([[-0.7, 0.3], [0.8, -0.5], [0.6, 0.2]]),
        ]

        for implementation in IMPLEMENTATIONS:
            with self.subTest(implementation=implementation.__name__):
                selector = implementation.MSelector(
                    hidden_dim=2, num_modalities=3, gumbel_tau=0.7)
                selector.train()

                # This is the pre-control training formula. Resetting the seed
                # makes both MLP dropout and Gumbel noise identical.
                torch.manual_seed(20260712)
                logits = selector.mlp(torch.cat(pooled, dim=-1))
                expected_primary = F.gumbel_softmax(
                    logits, tau=selector.gumbel_tau, hard=True)
                expected_weights = F.softmax(
                    logits / selector.gumbel_tau, dim=-1)
                expected_idx = torch.argmax(expected_primary, dim=-1)

                torch.manual_seed(20260712)
                weights, primary, primary_idx = selector._route(pooled)

                self.assertTrue(torch.equal(weights, expected_weights))
                self.assertTrue(torch.equal(primary, expected_primary))
                self.assertTrue(torch.equal(primary_idx, expected_idx))

                model = implementation.InfoGate(_tiny_args())
                confidence = torch.rand(2, 3, 2)
                mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
                self.assertEqual(model.confidence_mode, 'token')
                self.assertEqual(model.routing_mode, 'gumbel')
                self.assertIs(
                    model._apply_confidence_mode(confidence, mask),
                    confidence,
                )

    def test_modality_and_uniform_confidence_mask_padding(self):
        confidence = torch.tensor([
            [[0.1, 0.2], [0.3, 0.8], [0.9, 0.7], [0.6, 0.4]],
            [[0.2, 0.4], [0.9, 0.1], [0.6, 0.8], [0.3, 0.5]],
        ])
        mask = torch.tensor([
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ])

        for implementation in IMPLEMENTATIONS:
            with self.subTest(
                    implementation=implementation.__name__,
                    confidence_mode='modality'):
                model = implementation.InfoGate(
                    _tiny_args(confidence_mode='modality'))
                actual = model._apply_confidence_mode(confidence, mask)

                expected = torch.zeros_like(confidence)
                expected[0, :2] = confidence[0, :2].mean()
                expected[1, 0] = confidence[1, [0, 2]].mean()
                expected[1, 2] = confidence[1, [0, 2]].mean()
                torch.testing.assert_close(actual, expected)
                self.assertTrue(torch.equal(
                    actual[mask == 0], torch.zeros_like(actual[mask == 0])))

            with self.subTest(
                    implementation=implementation.__name__,
                    confidence_mode='uniform'):
                model = implementation.InfoGate(
                    _tiny_args(confidence_mode='uniform'))
                actual = model._apply_confidence_mode(confidence, mask)

                expected = torch.zeros_like(confidence)
                expected[0, :2] = 0.5
                expected[1, 0] = 0.5
                expected[1, 2] = 0.5
                torch.testing.assert_close(actual, expected)
                self.assertTrue(torch.equal(
                    actual[mask == 0], torch.zeros_like(actual[mask == 0])))

    def test_soft_and_argmax_routing(self):
        base_logits = torch.tensor([
            [2.0, 0.5, -1.0],
            [-0.5, 1.5, 0.2],
        ])

        for implementation in IMPLEMENTATIONS:
            with self.subTest(
                    implementation=implementation.__name__,
                    routing_mode='soft'):
                logits = base_logits.clone().requires_grad_()
                selector = implementation.MSelector(
                    hidden_dim=1, num_modalities=3, gumbel_tau=0.5,
                    routing_mode='soft')
                selector.mlp = nn.Identity()
                selector.train()

                weights, primary, primary_idx = selector._route([
                    logits[:, 0:1], logits[:, 1:2], logits[:, 2:3],
                ])
                expected_weights = F.softmax(logits / 0.5, dim=-1)
                torch.testing.assert_close(weights, expected_weights)
                self.assertIs(primary, weights)
                self.assertTrue(torch.equal(
                    primary_idx, expected_weights.argmax(dim=-1)))

                model = implementation.InfoGate(
                    _tiny_args(routing_mode='soft'))
                bottlenecks = [
                    torch.full((2, 1, 1), 1.0),
                    torch.full((2, 1, 1), 3.0),
                    torch.full((2, 1, 1), 5.0),
                ]
                confidences = [torch.ones_like(x) for x in bottlenecks]
                mixed, _, _, _ = model._route_by_primary(
                    bottlenecks, confidences,
                    weights, primary, primary_idx)
                expected_mixed = (
                    torch.stack(bottlenecks, dim=1)
                    * expected_weights[:, :, None, None]
                ).sum(dim=1)
                torch.testing.assert_close(mixed, expected_mixed)
                mixed.sum().backward()
                self.assertIsNotNone(logits.grad)

            with self.subTest(
                    implementation=implementation.__name__,
                    routing_mode='argmax'):
                logits = base_logits.clone().requires_grad_()
                selector = implementation.MSelector(
                    hidden_dim=1, num_modalities=3, gumbel_tau=0.5,
                    routing_mode='argmax')
                selector.mlp = nn.Identity()
                selector.train()

                weights, primary, primary_idx = selector._route([
                    logits[:, 0:1], logits[:, 1:2], logits[:, 2:3],
                ])
                expected_weights = F.softmax(logits / 0.5, dim=-1)
                expected_idx = expected_weights.argmax(dim=-1)
                expected_primary = F.one_hot(
                    expected_idx, num_classes=3).float()
                torch.testing.assert_close(weights, expected_weights)
                self.assertTrue(torch.equal(primary, expected_primary))
                self.assertTrue(torch.equal(primary_idx, expected_idx))
                self.assertTrue(weights.requires_grad)
                self.assertFalse(primary.requires_grad)

                # The hard decision remains hard and detached in eval mode.
                selector.eval()
                eval_weights, eval_primary, eval_idx = selector._route([
                    logits[:, 0:1], logits[:, 1:2], logits[:, 2:3],
                ])
                self.assertTrue(torch.equal(
                    eval_primary,
                    F.one_hot(eval_weights.argmax(dim=-1), 3).float()))
                self.assertTrue(torch.equal(
                    eval_idx, eval_weights.argmax(dim=-1)))
                self.assertFalse(eval_primary.requires_grad)

    def test_batch_gumbel_is_shared(self):
        logits = torch.tensor([
            [5.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 3.0],
        ])

        for implementation in IMPLEMENTATIONS:
            with self.subTest(implementation=implementation.__name__):
                selector = implementation.MSelector(
                    hidden_dim=1, num_modalities=3, gumbel_tau=0.8,
                    routing_mode='batch_gumbel')
                selector.mlp = nn.Identity()
                selector.train()
                shared_logits = logits.mean(dim=0, keepdim=True)

                torch.manual_seed(314159)
                expected_primary = F.gumbel_softmax(
                    shared_logits, tau=0.8, hard=True)
                expected_weights = F.softmax(logits / 0.8, dim=-1)

                torch.manual_seed(314159)
                weights, primary, primary_idx = selector._route([
                    logits[:, 0:1], logits[:, 1:2], logits[:, 2:3],
                ])
                torch.testing.assert_close(weights, expected_weights)
                self.assertTrue(torch.equal(
                    primary, expected_primary.expand_as(primary)))
                self.assertTrue(torch.equal(
                    primary_idx, primary_idx[0].expand_as(primary_idx)))
                self.assertEqual(tuple(weights.shape), (3, 3))
                self.assertEqual(tuple(primary_idx.shape), (3,))

                selector.eval()
                weights, primary, primary_idx = selector._route([
                    logits[:, 0:1], logits[:, 1:2], logits[:, 2:3],
                ])
                expected_weights = F.softmax(logits, dim=-1)
                expected_idx = shared_logits.argmax(dim=-1)
                expected_primary = F.one_hot(expected_idx, 3).float()
                torch.testing.assert_close(weights, expected_weights)
                self.assertTrue(torch.equal(
                    primary, expected_primary.expand_as(primary)))
                self.assertTrue(torch.equal(
                    primary_idx, expected_idx.expand_as(primary_idx)))

    def test_three_and_four_modality_forward_shapes(self):
        batch_size, seq_len = 2, 3
        text = torch.randn(batch_size, seq_len, 2)
        acoustic = torch.randn(batch_size, seq_len, 2)
        visual = torch.randn(batch_size, seq_len, 2)
        hcf = torch.randn(batch_size, seq_len, 4)
        labels = torch.tensor([0.0, 1.0])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        for implementation in IMPLEMENTATIONS:
            for hcf_dim in (0, 4):
                with self.subTest(
                        implementation=implementation.__name__,
                        hcf_dim=hcf_dim):
                    model = implementation.InfoGate(_tiny_args(
                        hcf_dim=hcf_dim,
                        task_type='binary',
                        confidence_mode='uniform',
                        routing_mode='batch_gumbel',
                    ))
                    model.eval()
                    logits, ib_loss, diagnostics, pooled = model(
                        text,
                        acoustic,
                        visual,
                        hcf=hcf if hcf_dim else None,
                        labels=labels,
                        stage=2,
                        attention_mask=mask,
                    )

                    self.assertEqual(tuple(logits.shape), (batch_size, 1))
                    self.assertEqual(tuple(pooled.shape), (batch_size, 2))
                    self.assertEqual(ib_loss.ndim, 0)
                    self.assertEqual(
                        'w_hcf' in diagnostics,
                        bool(hcf_dim),
                    )
                    self.assertEqual(
                        'primary_h' in diagnostics,
                        bool(hcf_dim),
                    )


if __name__ == '__main__':
    unittest.main()
