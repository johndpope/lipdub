import torch

from custom_types import *


def vs_to_affine(vs: T, required_dim=3) -> T:
    # if vs.shape[-1] == required_dim:
    return torch.cat([vs, torch.ones(*vs.shape[:-1], 1, device=vs.device, dtype=vs.dtype)], dim=-1)
    # return vs


def vs_to_mat(vs: T) -> T:
    num_mats, num_vs, dim = vs.shape
    vs_affine = vs_to_affine(vs)
    blocks = [torch.zeros(num_mats, num_vs * dim, dim + 1,  device=vs.device, dtype=vs.dtype) for _ in range(dim)]
    arange = torch.arange(num_vs, device=vs.device) * dim
    for i in range(dim):
        blocks[i][:, arange + i] = vs_affine
    return torch.cat(blocks, dim=-1)


def to_affine(transformation: T, required_dim=3) -> T:
    if transformation.dim() >= 2 and transformation.shape[-1] == required_dim + 1:
        return transformation
    affine = torch.eye(required_dim + 1)
    if transformation.dim() == 2:
        affine[:required_dim, :required_dim] = transformation
    else:
        affine[:required_dim, required_dim] = transformation
    return affine


def apply_affine(affine: T, vs: T) -> T:
    dim = vs.shape[-1]
    # should_reduce = dim == required_dim
    affine = to_affine(affine, dim)
    vs = vs_to_affine(vs)
    if affine.dim() == 3:
        vs_transformed = torch.einsum('bad,bnd->bna', [affine, vs])
        vs_transformed = vs_transformed[:, :, :dim]
    else:
        vs_transformed = torch.einsum('ad,nd->na', [affine, vs])
        vs_transformed = vs_transformed[:, :dim]
    return vs_transformed


def find_affine(vs_source: T, vs_target: T) -> T:
    device = vs_source.device
    should_reduce = vs_source.dim() == 2
    if should_reduce:
        vs_source, vs_target = vs_source.unsqueeze(0), vs_target.unsqueeze(0)
    num_groups, _, dim = vs_source.shape
    assert vs_source.shape[1] > dim and vs_source.shape == vs_target.shape
    vs_source_mat = vs_to_mat(vs_source)
    affine = vs_source_mat.pinverse().matmul(vs_target.view(num_groups, -1, 1))
    affine = affine.view(num_groups, dim, dim + 1)
    affine_row = torch.zeros(num_groups, 1, dim + 1, device=device, dtype=vs_source.dtype)
    affine_row[:, 0, -1] = 1
    affine = torch.cat((affine, affine_row), dim=1)
    if should_reduce:
        affine = affine.squeeze(0)
    return affine


def align_landmarks(source_lm: ARRAY, target_lm: ARRAY) -> ARRAY:
    source_lm_t = torch.from_numpy(source_lm)
    target_lm_t: T = torch.from_numpy(target_lm)
    select = np.concatenate([np.arange(17), 36 + np.arange(12)])  # jaw and eyes
    if target_lm_t.dim() == 3:
        affine = find_affine(source_lm_t[:, select], target_lm_t[:, select])
    else:
        affine = find_affine(source_lm_t[select], target_lm_t[select])
    source_lm_transformed = apply_affine(affine, source_lm_t)
    return source_lm_transformed.numpy()


if __name__ == '__main__':
    source = np.random.rand(68, 2)
    target = np.random.rand(68, 2)
    target = align_landmarks(source, target)
