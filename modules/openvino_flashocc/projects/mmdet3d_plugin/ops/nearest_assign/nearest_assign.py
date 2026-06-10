"""Nearest assign operation with CUDA extension and CPU fallback."""
import torch
import warnings

try:
    from . import nearest_assign_ext
    _HAS_CUDA_EXT = True
except ImportError:
    nearest_assign_ext = None
    _HAS_CUDA_EXT = False
    warnings.warn(
        "nearest_assign CUDA extension not available. Falling back to PyTorch implementation.",
        stacklevel=2,
    )


def _nearest_assign_fallback(
    occ_pred,
    l2s_key,
    occind2detind,
    inst_cls,
    inst_xyz,
    inst_id_list,
):
    """Fallback that mirrors nearest_assign CUDA kernel semantics.

    Semantics replicated from nearest_assign_cuda.cu:
      - If voxel label is in l2s_key:
          find nearest instance with class == occind2detind[label]
          assign inst_pred[x, y, z] = inst_id_list[nearest]
      - Else:
          inst_pred[x, y, z] = occ_pred[x, y, z]
    """
    occ_pred = occ_pred.contiguous().int()
    l2s_key = l2s_key.contiguous().int().view(-1)
    occind2detind = occind2detind.contiguous().int().view(-1)
    inst_cls = inst_cls.contiguous().int().view(-1)
    inst_xyz = inst_xyz.contiguous().int().view(-1, 3)
    inst_id_list = inst_id_list.contiguous().int().view(-1)

    inst_pred = occ_pred.clone()

    if occ_pred.dim() != 3:
        raise ValueError(f"occ_pred must be 3D [X, Y, Z], got shape={tuple(occ_pred.shape)}")

    nx, ny, nz = occ_pred.shape

    if l2s_key.numel() == 0 or inst_cls.numel() == 0:
        return inst_pred

    labels_to_process = set(int(v) for v in l2s_key.detach().cpu().tolist())

    for occ_label in labels_to_process:
        if occ_label < 0 or occ_label >= occind2detind.numel():
            continue

        voxel_mask = occ_pred == occ_label
        if not voxel_mask.any():
            continue

        det_label = int(occind2detind[occ_label].item())
        inst_mask = inst_cls == det_label

        # Match CUDA kernel behavior: if no matching instance exists,
        # values for processed voxels are left as initialized zeros in kernel.
        # Here we mirror that by writing 0 for this subset.
        if not inst_mask.any():
            inst_pred[voxel_mask] = 0
            continue

        candidate_xyz = inst_xyz[inst_mask].to(torch.long)
        candidate_ids = inst_id_list[inst_mask]

        voxel_coords = voxel_mask.nonzero(as_tuple=False).to(torch.long)

        # squared L2 distance, same criterion as CUDA kernel
        diffs = voxel_coords[:, None, :] - candidate_xyz[None, :, :]
        dists = (diffs * diffs).sum(dim=-1)
        nearest_idx = dists.argmin(dim=1)
        inst_pred[voxel_mask] = candidate_ids[nearest_idx]

    return inst_pred


def nearest_assign(
    occ_pred,
    l2s_key,
    occind2detind,
    inst_cls,
    inst_xyz,
    inst_id_list,
):
    """Nearest assign with original 6-argument interface.

    Args:
        occ_pred (Tensor): occupancy prediction labels, shape [X, Y, Z]
        l2s_key (Tensor): occ labels that require instance assignment
        occind2detind (Tensor): mapping from occ-label to det-label
        inst_cls (Tensor): instance detection labels
        inst_xyz (Tensor): instance centers in voxel coordinates, shape [N, 3]
        inst_id_list (Tensor): instance IDs written into output volume

    Returns:
        Tensor: assigned instance volume, same shape as occ_pred
    """
    # CUDA fast path (original behavior)
    if _HAS_CUDA_EXT and occ_pred.is_cuda:
        occ_pred = occ_pred.contiguous().int()
        l2s_key = l2s_key.contiguous().int()
        occind2detind = occind2detind.contiguous().int()
        inst_cls = inst_cls.contiguous().int()
        inst_xyz = inst_xyz.contiguous().int()
        inst_id_list = inst_id_list.contiguous().int()
        inst_pred = occ_pred.new_zeros(occ_pred.shape)
        nearest_assign_ext.nearest_assign_forward(
            occ_pred,
            l2s_key,
            occind2detind,
            inst_cls,
            inst_xyz,
            inst_id_list,
            inst_pred,
        )
        return inst_pred

    # CPU/OpenVINO-safe fallback path
    return _nearest_assign_fallback(
        occ_pred,
        l2s_key,
        occind2detind,
        inst_cls,
        inst_xyz,
        inst_id_list,
    )


__all__ = ["nearest_assign"]
