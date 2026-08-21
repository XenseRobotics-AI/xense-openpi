"""Offline tactile counterfactual inference diagnostic tool.

Reads a YAML probe spec (pairs of (episode, frame) full/empty samples), runs
four inference conditions per pair (F_F / F_E / E_E / E_F) with paired flow
noise, and saves per-run traces (FastViT features, tactile tokens, suffix
info, action hidden states, v_t, x_t) plus sensitivity metrics.

All model math is performed by a trace-equivalent copy of the production
sampler (``trace_sampler.trace_sample_actions``) that must be verified
allclose against ``Pi0.sample_actions`` / ``Pi0.training_time_rtc_sample_actions``
before any experiment run is allowed to proceed.
"""
