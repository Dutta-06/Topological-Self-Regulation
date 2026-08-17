"""TSR-X Structural Edit Engine (Definition 5.3 & Remark 5.4).

Executes exact by-index mutations across coupled index bundles:
  - prune_group_index: Slices out channel index j across all producer, affine,
    and consumer slots in the group without disturbing surviving indices.
  - materialize_candidate: Promotes a candidate channel to real status, setting
    outgoing port weights along the optimal gradient descent direction v* (Theorem 3.4)
    and refilling a fresh candidate.
  - reindex_optimizer_state: In-place slicing / expanding of optimizer state tensors
    (momentum, Adam exp_avg, exp_avg_sq) so learning proceeds smoothly without shock.
"""

from typing import Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn

from tsrx.graph.bundle import IndexBundle, ParamSlot
from tsrx.sense.candidates import CandidateBank


def _modules_by_name(model: nn.Module) -> Dict[str, nn.Module]:
    return dict(model.named_modules())


def _bump_out_attr(mod: nn.Module, delta: int) -> None:
    for attr in ("out_channels", "out_features"):
        if hasattr(mod, attr):
            setattr(mod, attr, getattr(mod, attr) + delta)


def _bump_in_attr(mod: nn.Module, delta: int) -> None:
    for attr in ("in_channels", "in_features"):
        if hasattr(mod, attr):
            setattr(mod, attr, getattr(mod, attr) + delta)


def reindex_optimizer_state(
    optimizer: torch.optim.Optimizer,
    old_param: nn.Parameter,
    new_param: nn.Parameter,
    axis: int,
    action: str,  # "prune" or "grow"
    idx: int,
    count: int = 1,
    fill_value: float = 0.0,
) -> None:
    """Update optimizer state dictionary for a resized parameter tensor."""
    if optimizer is None:
        return

    # Find the parameter in optimizer state
    state = optimizer.state.get(old_param)
    if state is None or len(state) == 0:
        return

    new_state = {}
    for k, v in state.items():
        if torch.is_tensor(v) and v.shape == old_param.shape:
            if action == "prune":
                # Slice out [idx : idx + count] along axis
                slices_before = [slice(None)] * axis + [slice(0, idx)]
                slices_after = [slice(None)] * axis + [slice(idx + count, None)]
                v_new = torch.cat([v[tuple(slices_before)], v[tuple(slices_after)]], dim=axis)
            elif action == "grow":
                # Insert count elements at idx along axis
                slices_before = [slice(None)] * axis + [slice(0, idx)]
                slices_after = [slice(None)] * axis + [slice(idx, None)]
                shape_inserted = list(v.shape)
                shape_inserted[axis] = count
                inserted = torch.full(shape_inserted, fill_value, device=v.device, dtype=v.dtype)
                v_new = torch.cat([v[tuple(slices_before)], inserted, v[tuple(slices_after)]], dim=axis)
            else:
                v_new = v
            new_state[k] = v_new
        else:
            new_state[k] = v

    # Swap in optimizer parameter list & state dict
    del optimizer.state[old_param]
    optimizer.state[new_param] = new_state

    for group in optimizer.param_groups:
        for i, p in enumerate(group["params"]):
            if p is old_param:
                group["params"][i] = new_param


def prune_group_index(
    model: nn.Module,
    bundle: IndexBundle,
    idx: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    bank: Optional[CandidateBank] = None,
) -> None:
    """Prune channel index `idx` across all blocks indexed by `bundle`.

    Preserves function if the pruned channel is null or low-saliency (Theorem 5.5).
    """
    assert 0 <= idx < bundle.size, f"Invalid index {idx} for bundle size {bundle.size}"
    modules = _modules_by_name(model)

    # 1. Prune Producer Slots (axis=0 of weight, bias)
    producers_seen = set()
    for slot in bundle.producer_slots:
        if slot.module_name in producers_seen:
            continue
        producers_seen.add(slot.module_name)
        mod = modules[slot.module_name]

        # Weight (out_channels, in_channels, *kernel)
        w = mod.weight
        w_new = torch.cat([w.data[:idx], w.data[idx + 1:]], dim=0)
        new_param = nn.Parameter(w_new)
        reindex_optimizer_state(optimizer, mod.weight, new_param, axis=0, action="prune", idx=idx)
        mod.weight = new_param

        # Bias (if present)
        if getattr(mod, "bias", None) is not None:
            b = mod.bias
            b_new = torch.cat([b.data[:idx], b.data[idx + 1:]], dim=0)
            new_b_param = nn.Parameter(b_new)
            reindex_optimizer_state(optimizer, mod.bias, new_b_param, axis=0, action="prune", idx=idx)
            mod.bias = new_b_param

        # Update module attribute
        if hasattr(mod, "out_channels"):
            mod.out_channels -= 1
        elif hasattr(mod, "out_features"):
            mod.out_features -= 1

    # 2. Prune Affine Slots (BatchNorm / InstanceNorm)
    affines_seen = set()
    for slot in bundle.affine_slots:
        if slot.module_name in affines_seen:
            continue
        affines_seen.add(slot.module_name)
        mod = modules[slot.module_name]

        for pname in ("weight", "bias"):
            if hasattr(mod, pname) and getattr(mod, pname) is not None:
                param = getattr(mod, pname)
                p_new = torch.cat([param.data[:idx], param.data[idx + 1:]], dim=0)
                new_param = nn.Parameter(p_new)
                reindex_optimizer_state(optimizer, param, new_param, axis=0, action="prune", idx=idx)
                setattr(mod, pname, new_param)

        for bname in ("running_mean", "running_var"):
            if hasattr(mod, bname) and getattr(mod, bname) is not None:
                buf = getattr(mod, bname)
                b_new = torch.cat([buf[:idx], buf[idx + 1:]], dim=0)
                setattr(mod, bname, b_new)

        for attr in ("num_features", "num_channels"):
            if hasattr(mod, attr):
                setattr(mod, attr, getattr(mod, attr) - 1)

    # 3. Prune Consumer Slots (axis=1 of weight, with multiplicity)
    consumers_seen = set()
    for slot in bundle.consumer_slots:
        if slot.module_name in consumers_seen:
            continue
        consumers_seen.add(slot.module_name)
        mod = modules[slot.module_name]

        w = mod.weight
        mult = slot.multiplicity
        start_col = idx * mult
        end_col = (idx + 1) * mult

        w_new = torch.cat([w.data[:, :start_col], w.data[:, end_col:]], dim=1)
        new_param = nn.Parameter(w_new)
        reindex_optimizer_state(optimizer, mod.weight, new_param, axis=1, action="prune", idx=start_col, count=mult)
        mod.weight = new_param

        if hasattr(mod, "in_channels"):
            mod.in_channels -= mult
        elif hasattr(mod, "in_features"):
            mod.in_features -= mult

    bundle.size -= 1
    if bank is not None and bundle.tap in bank.handles:
        bank.handles[bundle.tap].base_size -= 1


def materialize_candidate(
    bank: CandidateBank,
    tap: int,
    cand_idx: int,
    eps: float = 1e-3,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """Materialize candidate `cand_idx` (0 <= cand_idx < k) into a real channel.

    1. Moves candidate `cand_idx` into the real channel set at index `base_size`.
    2. Initializes consumer port weights along optimal descent direction v* = -eps * u_c / ||u_c|| (Theorem 3.4).
    3. Expands tensors so exactly `k` candidate slots remain available for subsequent steps.
    4. Increments base_size and bundle.size.
    """
    h = bank.handles[tap]
    model = bank.model
    modules = _modules_by_name(model)
    k = h.k
    base_size = h.base_size
    cand_slot_idx = base_size + cand_idx

    # 1. Expand Producer Slots
    producers_seen = set()
    for slot in h.bundle.producer_slots:
        if slot.module_name in producers_seen:
            continue
        producers_seen.add(slot.module_name)
        mod = modules[slot.module_name]

        w = mod.weight
        device, dtype = w.device, w.dtype
        real_w = w.data[:base_size]
        promoted_w = w.data[cand_slot_idx : cand_slot_idx + 1].clone()
        cands_w = w.data[base_size:].clone()

        # Refill candidate at cand_idx with new Kaiming weights
        new_cand_row = torch.empty(1, *w.shape[1:], device=device, dtype=dtype)
        if new_cand_row.numel():
            nn.init.kaiming_uniform_(new_cand_row.view(1, -1), a=math.sqrt(5))
        cands_w[cand_idx] = new_cand_row[0]

        w_new = torch.cat([real_w, promoted_w, cands_w], dim=0)
        new_param = nn.Parameter(w_new)
        reindex_optimizer_state(optimizer, mod.weight, new_param, axis=0, action="grow", idx=base_size, count=1)
        mod.weight = new_param

        # Bias
        if getattr(mod, "bias", None) is not None:
            b = mod.bias
            real_b = b.data[:base_size]
            promoted_b = b.data[cand_slot_idx : cand_slot_idx + 1].clone()
            cands_b = b.data[base_size:].clone()
            cands_b[cand_idx] = 0.0
            b_new = torch.cat([real_b, promoted_b, cands_b], dim=0)
            new_b = nn.Parameter(b_new)
            reindex_optimizer_state(optimizer, mod.bias, new_b, axis=0, action="grow", idx=base_size, count=1)
            mod.bias = new_b

        _bump_out_attr(mod, 1)

    # 2. Expand Affine Slots (BatchNorm)
    affines_seen = set()
    for slot in h.bundle.affine_slots:
        if slot.module_name in affines_seen:
            continue
        affines_seen.add(slot.module_name)
        mod = modules[slot.module_name]
        device = mod.weight.device if getattr(mod, "weight", None) is not None else "cpu"
        dtype = mod.weight.dtype if getattr(mod, "weight", None) is not None else torch.float32

        if getattr(mod, "weight", None) is not None:
            pw = mod.weight
            real_pw = pw.data[:base_size]
            promoted_pw = pw.data[cand_slot_idx : cand_slot_idx + 1].clone()
            cands_pw = pw.data[base_size:].clone()
            cands_pw[cand_idx] = 1.0
            pw_new = torch.cat([real_pw, promoted_pw, cands_pw], dim=0)
            new_param = nn.Parameter(pw_new)
            reindex_optimizer_state(optimizer, mod.weight, new_param, axis=0, action="grow", idx=base_size, count=1)
            mod.weight = new_param

        if getattr(mod, "bias", None) is not None:
            pb = mod.bias
            real_pb = pb.data[:base_size]
            promoted_pb = pb.data[cand_slot_idx : cand_slot_idx + 1].clone()
            cands_pb = pb.data[base_size:].clone()
            cands_pb[cand_idx] = 0.0
            pb_new = torch.cat([real_pb, promoted_pb, cands_pb], dim=0)
            new_param = nn.Parameter(pb_new)
            reindex_optimizer_state(optimizer, mod.bias, new_param, axis=0, action="grow", idx=base_size, count=1)
            mod.bias = new_param

        if getattr(mod, "running_mean", None) is not None:
            rm = mod.running_mean
            real_rm = rm[:base_size]
            promoted_rm = rm[cand_slot_idx : cand_slot_idx + 1].clone()
            cands_rm = rm[base_size:].clone()
            cands_rm[cand_idx] = 0.0
            mod.running_mean = torch.cat([real_rm, promoted_rm, cands_rm], dim=0)

        if getattr(mod, "running_var", None) is not None:
            rv = mod.running_var
            real_rv = rv[:base_size]
            promoted_rv = rv[cand_slot_idx : cand_slot_idx + 1].clone()
            cands_rv = rv[base_size:].clone()
            cands_rv[cand_idx] = 1.0
            mod.running_var = torch.cat([real_rv, promoted_rv, cands_rv], dim=0)

        for attr in ("num_features", "num_channels"):
            if hasattr(mod, attr):
                setattr(mod, attr, getattr(mod, attr) + 1)

    # 3. Expand Consumer Slots and Wire Port Direction v* = -eps * u_c / ||u_c||
    consumers_seen = set()
    for slot in h.bundle.consumer_slots:
        if slot.module_name in consumers_seen:
            continue
        consumers_seen.add(slot.module_name)
        mod = modules[slot.module_name]

        mult = slot.multiplicity
        w = mod.weight
        device, dtype = w.device, w.dtype
        g = mod.weight.grad

        real_cols = w.data[:, : base_size * mult]
        cand_cols = slice(cand_slot_idx * mult, (cand_slot_idx + 1) * mult)

        if g is not None:
            cand_g = g.data[:, cand_cols]
            norm = cand_g.norm().clamp_min(1e-12)
            v_star = -eps * (cand_g / norm)
        else:
            v_star = torch.empty(w.shape[0], mult, *w.shape[2:], device=device, dtype=dtype)
            nn.init.kaiming_uniform_(v_star.reshape(w.shape[0], -1), a=math.sqrt(5))
            v_star *= eps

        cands_all = w.data[:, base_size * mult :].clone()
        # Zero out the refilled candidate slot
        cands_all[:, cand_idx * mult : (cand_idx + 1) * mult] = 0.0

        w_new = torch.cat([real_cols, v_star, cands_all], dim=1)
        new_param = nn.Parameter(w_new)
        reindex_optimizer_state(optimizer, mod.weight, new_param, axis=1, action="grow", idx=base_size * mult, count=mult)
        mod.weight = new_param

        _bump_in_attr(mod, mult)

    # 4. Increment sizes
    h.base_size += 1
    h.bundle.size += 1
