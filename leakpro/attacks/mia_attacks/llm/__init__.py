#
# Copyright 2023-2026 Lindholmen Science Park AB
# SPDX-License-Identifier: Apache-2.0
#
"""Membership-inference attacks against fine-tuned autoregressive language models.

One module per attack. All of them build on :class:`~leakpro.attacks.mia_attacks.llm.abstract_llm_mia.AbstractLLMMIA`,
which turns the audit indices into per-token :class:`~leakpro.signals.token_evidence.TokenEvidence`
for the target and any frozen reference models; each attack is then a NumPy reduction over
those arrays.
"""
