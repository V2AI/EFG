# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn.functional as F
from torch.autograd import Function

from efg import _C

# -------------------------------------------------- #
#                  CONSTANTS                         #
# -------------------------------------------------- #
"""
_box_planes and _box_triangles define the 4- and 3-connectivity
of the 8 box corners.
_box_planes gives the quad faces of the 3D box
_box_triangles gives the triangle faces of the 3D box
"""
_box_planes = [
    [0, 1, 2, 3],
    [3, 2, 6, 7],
    [0, 1, 5, 4],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
    [4, 5, 6, 7],
]
_box_triangles = [
    [0, 1, 2],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [1, 5, 6],
    [1, 6, 2],
    [0, 4, 7],
    [0, 7, 3],
    [3, 2, 6],
    [3, 6, 7],
    [0, 1, 5],
    [0, 4, 5],
]

DOT_EPS = 1e-3
AREA_EPS = 1e-4


def _check_coplanar(boxes: torch.Tensor, eps: float = 1e-8) -> None:
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)
    if not (mat1.bmm(mat2).abs() < eps).all().item():
        msg = "Plane vertices are not coplanar"
        raise ValueError(msg)

    return


def _check_nonzero(boxes: torch.Tensor, eps: float = 1e-8) -> None:
    """
    Checks that the sides of the box have a non zero area
    """
    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2

    if (face_areas < eps).any().item():
        msg = "Planes have zero areas"
        raise ValueError(msg)

    return


class _box3d_overlap(Function):
    """
    Torch autograd Function wrapper for box3d_overlap C++/CUDA implementations.
    Backward is not supported.
    """

    @staticmethod
    def forward(ctx, boxes1, boxes2):
        """
        Arguments defintions the same as in the box3d_overlap function
        """
        vol, iou = _C.iou_box3d(boxes1, boxes2)
        return vol, iou

    @staticmethod
    def backward(ctx, grad_vol, grad_iou):
        raise ValueError("box3d_overlap backward is not supported")


def box3d_overlap(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the intersection of 3D boxes1 and boxes2.
    Inputs boxes1, boxes2 are tensors of shape (B, 8, 3)
    (where B doesn't have to be the same for boxes1 and boxes1),
    containing the 8 corners of the boxes, as follows:
        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)
    NOTE: Throughout this implementation, we assume that boxes
    are defined by their 8 corners exactly in the order specified in the
    diagram above for the function to give correct results. In addition
    the vertices on each plane must be coplanar.
    As an alternative to the diagram, this is a unit bounding
    box which has the correct vertex ordering:
    box_corner_vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
    Args:
        boxes1: tensor of shape (N, 8, 3) of the coordinates of the 1st boxes
        boxes2: tensor of shape (M, 8, 3) of the coordinates of the 2nd boxes
    Returns:
        vol: (N, M) tensor of the volume of the intersecting convex shapes
        iou: (N, M) tensor of the intersection over union which is
            defined as: `iou = vol / (vol1 + vol2 - vol)`
    """
    if not all((8, 3) == box.shape[1:] for box in [boxes1, boxes2]):
        raise ValueError("Each box in the batch must be of shape (8, 3)")

    _check_coplanar(boxes1, eps)
    _check_coplanar(boxes2, eps)
    _check_nonzero(boxes1, eps)
    _check_nonzero(boxes2, eps)

    # pyre-fixme[16]: `_box3d_overlap` has no attribute `apply`.
    vol, iou = _box3d_overlap.apply(boxes1, boxes2)

    return vol, iou


def box3d_overlap_sampling_batched(boxes1, boxes2, num_samples: int):
    """
    Wrapper around box3d_overlap_sampling to support
    batched input
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    ious = torch.zeros((N, M), dtype=torch.float32, device=boxes1.device)
    for n in range(N):
        for m in range(M):
            iou = box3d_overlap_sampling(boxes1[n], boxes2[m])
            ious[n, m] = iou
    return ious


def box3d_overlap_sampling(box1: torch.Tensor, box2: torch.Tensor, num_samples: int = 10000):
    """
    Computes the intersection of two boxes by sampling points
    """
    vol1 = box_volume(box1)
    vol2 = box_volume(box2)

    points1 = sample_points_within_box(box1, num_samples=num_samples)
    points2 = sample_points_within_box(box2, num_samples=num_samples)

    isin21 = is_point_inside_box(box1, points2)
    num21 = isin21.sum()
    isin12 = is_point_inside_box(box2, points1)
    num12 = isin12.sum()

    assert num12 <= num_samples
    assert num21 <= num_samples

    inters = (vol1 * num12 + vol2 * num21) / 2.0
    union = vol1 * num_samples + vol2 * num_samples - inters
    return inters / union


def box_volume(box: torch.Tensor) -> torch.Tensor:
    """
    Computes the volume of each box in boxes.
    The volume of each box is the sum of all the tetrahedrons
    formed by the faces of the box. The face of the box is the base of
    that tetrahedron and the center point of the box is the apex.
    In other words, vol(box) = sum_i A_i * d_i / 3,
    where A_i is the area of the i-th face and d_i is the
    distance of the apex from the face.
    We use the equivalent dot/cross product formulation.
    Read https://en.wikipedia.org/wiki/Tetrahedron#Volume
    Args:
        box: tensor of shape (8, 3) containing the vertices
            of the 3D box
    Returns:
        vols: the volume of the box
    """
    assert box.shape[0] == 8 and box.shape[1] == 3

    # Compute the center point of each box
    ctr = box.mean(0).view(1, 1, 3)

    # Extract the coordinates of the faces for each box
    tri_verts = get_tri_verts(box)
    # Set the origin of the coordinate system to coincide
    # with the apex of the tetrahedron to simplify the volume calculation
    # See https://en.wikipedia.org/wiki/Tetrahedron#Volume
    tri_verts = tri_verts - ctr

    # Compute the volume of each box using the dot/cross product formula
    vols = torch.sum(
        tri_verts[:, 0] * torch.cross(tri_verts[:, 1], tri_verts[:, 2], dim=-1),
        dim=-1,
    )
    vols = (vols.abs() / 6.0).sum()

    return vols


def get_tri_verts(box: torch.Tensor) -> torch.Tensor:
    """
    Return the vertex coordinates forming the triangles of the box.
    The computation here resembles the Meshes data structure.
    But since we only want this tiny functionality, we abstract it out.
    Args:
        box: tensor of shape (8, 3)
    Returns:
        tri_verts: tensor of shape (12, 3, 3)
    """
    device = box.device
    faces = torch.tensor(_box_triangles, device=device, dtype=torch.int64)  # (12, 3)
    tri_verts = box[faces]  # (12, 3, 3)
    return tri_verts


def sample_points_within_box(box: torch.Tensor, num_samples: int = 10):
    """
    Sample points within a box defined by its 8 coordinates
    Args:
        box: tensor of shape (8, 3) of the box coordinates
        num_samples: int defining the number of samples
    Returns:
        points: (num_samples, 3) of points inside the box
    """
    assert box.shape[0] == 8 and box.shape[1] == 3
    xyzmin = box.min(0).values.view(1, 3)
    xyzmax = box.max(0).values.view(1, 3)

    uvw = torch.rand((num_samples, 3), device=box.device)
    points = uvw * (xyzmax - xyzmin) + xyzmin

    # because the box is not axis aligned we need to check wether
    # the points are within the box
    num_points = 0
    samples = []
    while num_points < num_samples:
        inside = is_point_inside_box(box, points)
        samples.append(points[inside].view(-1, 3))
        num_points += inside.sum()

    samples = torch.cat(samples, dim=0)
    return samples[1:num_samples]


def is_point_inside_box(box: torch.Tensor, points: torch.Tensor):
    """
    Determines whether points are inside the boxes
    Args:
        box: tensor of shape (8, 3) of the corners of the boxes
        points: tensor of shape (P, 3) of the points
    Returns:
        inside: bool tensor of shape (P,)
    """
    device = box.device
    P = points.shape[0]

    n = box_planar_dir(box)  # (6, 3)
    box_planes = get_plane_verts(box)  # (6, 4)
    num_planes = box_planes.shape[0]  # = 6

    # a point p is inside the box if it "inside" all planes of the box
    # so we run the checks
    ins = torch.zeros((P, num_planes), device=device, dtype=torch.bool)
    for i in range(num_planes):
        is_in, _ = is_inside(box_planes[i], n[i], points, return_proj=False)
        ins[:, i] = is_in
    ins = ins.all(dim=1)
    return ins


def box_planar_dir(box: torch.Tensor, dot_eps: float = DOT_EPS, area_eps: float = AREA_EPS) -> torch.Tensor:
    """
    Finds the unit vector n which is perpendicular to each plane in the box
    and points towards the inside of the box.
    The planes are defined by `_box_planes`.
    Since the shape is convex, we define the interior to be the direction
    pointing to the center of the shape.
    Args:
       box: tensor of shape (8, 3) of the vertices of the 3D box
    Returns:
       n: tensor of shape (6,) of the unit vector orthogonal to the face pointing
          towards the interior of the shape
    """
    assert box.shape[0] == 8 and box.shape[1] == 3

    # center point of each box
    box_ctr = box.mean(0).view(1, 3)

    # box planes
    plane_verts = get_plane_verts(box)  # (6, 4, 3)
    v0, v1, v2, v3 = plane_verts.unbind(1)
    plane_ctr, n = get_plane_center_normal(plane_verts)

    # Check all verts are coplanar
    if not (F.normalize(v3 - v0, dim=-1).unsqueeze(1).bmm(n.unsqueeze(2)).abs() < dot_eps).all().item():
        msg = "Plane vertices are not coplanar"
        raise ValueError(msg)

    # Check all faces have non zero area
    area1 = torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1) / 2
    area2 = torch.cross(v3 - v0, v2 - v0, dim=-1).norm(dim=-1) / 2
    if (area1 < area_eps).any().item() or (area2 < area_eps).any().item():
        msg = "Planes have zero areas"
        raise ValueError(msg)

    # We can write:  `box_ctr = plane_ctr + a * e0 + b * e1 + c * n`, (1).
    # With <e0, n> = 0 and <e1, n> = 0, where <.,.> refers to the dot product,
    # since that e0 is orthogonal to n. Same for e1.
    """
    # Below is how one would solve for (a, b, c)
    # Solving for (a, b)
    numF = verts.shape[0]
    A = torch.ones((numF, 2, 2), dtype=torch.float32, device=device)
    B = torch.ones((numF, 2), dtype=torch.float32, device=device)
    A[:, 0, 1] = (e0 * e1).sum(-1)
    A[:, 1, 0] = (e0 * e1).sum(-1)
    B[:, 0] = ((box_ctr - plane_ctr) * e0).sum(-1)
    B[:, 1] = ((box_ctr - plane_ctr) * e1).sum(-1)
    ab = torch.linalg.solve(A, B)  # (numF, 2)
    a, b = ab.unbind(1)
    # solving for c
    c = ((box_ctr - plane_ctr - a.view(numF, 1) * e0 - b.view(numF, 1) * e1) * n).sum(-1)
    """
    # Since we know that <e0, n> = 0 and <e1, n> = 0 (e0 and e1 are orthogonal to n),
    # the above solution is equivalent to
    direc = F.normalize(box_ctr - plane_ctr, dim=-1)  # (6, 3)
    c = (direc * n).sum(-1)
    # If c is negative, then we revert the direction of n such that n points "inside"
    negc = c < 0.0
    n[negc] *= -1.0
    # c[negc] *= -1.0
    # Now (a, b, c) is the solution to (1)

    return n


def get_plane_verts(box: torch.Tensor) -> torch.Tensor:
    """
    Return the vertex coordinates forming the planes of the box.
    The computation here resembles the Meshes data structure.
    But since we only want this tiny functionality, we abstract it out.
    Args:
        box: tensor of shape (8, 3)
    Returns:
        plane_verts: tensor of shape (6, 4, 3)
    """
    device = box.device
    faces = torch.tensor(_box_planes, device=device, dtype=torch.int64)  # (6, 4)
    plane_verts = box[faces]  # (6, 4, 3)
    return plane_verts


def get_plane_center_normal(planes: torch.Tensor) -> torch.Tensor:
    """
    Returns the center and normal of planes
    Args:
        planes: tensor of shape (P, 4, 3)
    Returns:
        center: tensor of shape (P, 3)
        normal: tensor of shape (P, 3)
    """
    add_dim0 = False
    if planes.ndim == 2:
        planes = planes.unsqueeze(0)
        add_dim0 = True

    ctr = planes.mean(1)  # (P, 3)
    normals = torch.zeros_like(ctr)

    v0, v1, v2, v3 = planes.unbind(1)  # 4 x (P, 3)

    # unvectorized solution
    P = planes.shape[0]
    for t in range(P):
        ns = torch.zeros((6, 3), device=planes.device)
        ns[0] = torch.cross(v0[t] - ctr[t], v1[t] - ctr[t], dim=-1)
        ns[1] = torch.cross(v0[t] - ctr[t], v2[t] - ctr[t], dim=-1)
        ns[2] = torch.cross(v0[t] - ctr[t], v3[t] - ctr[t], dim=-1)
        ns[3] = torch.cross(v1[t] - ctr[t], v2[t] - ctr[t], dim=-1)
        ns[4] = torch.cross(v1[t] - ctr[t], v3[t] - ctr[t], dim=-1)
        ns[5] = torch.cross(v2[t] - ctr[t], v3[t] - ctr[t], dim=-1)

        i = torch.norm(ns, dim=-1).argmax()
        normals[t] = ns[i]

    if add_dim0:
        ctr = ctr[0]
        normals = normals[0]
    normals = F.normalize(normals, dim=-1)
    return ctr, normals


def is_inside(
    plane: torch.Tensor,
    n: torch.Tensor,
    points: torch.Tensor,
    return_proj: bool = True,
):
    """
    Computes whether point is "inside" the plane.
    The definition of "inside" means that the point
    has a positive component in the direction of the plane normal defined by n.
    For example,
                  plane
                    |
                    |         . (A)
                    |--> n
                    |
         .(B)       |
    Point (A) is "inside" the plane, while point (B) is "outside" the plane.
    Args:
      plane: tensor of shape (4,3) of vertices of a box plane
      n: tensor of shape (3,) of the unit "inside" direction on the plane
      points: tensor of shape (P, 3) of coordinates of a point
      return_proj: bool whether to return the projected point on the plane
    Returns:
      is_inside: bool of shape (P,) of whether point is inside
      p_proj: tensor of shape (P, 3) of the projected point on plane
    """
    device = plane.device
    v0, v1, v2, v3 = plane.unbind(0)
    plane_ctr = plane.mean(0)
    e0 = F.normalize(v0 - plane_ctr, dim=0)
    e1 = F.normalize(v1 - plane_ctr, dim=0)
    if not torch.allclose(e0.dot(n), torch.zeros((1,), device=device), atol=1e-2):
        raise ValueError("Input n is not perpendicular to the plane")
    if not torch.allclose(e1.dot(n), torch.zeros((1,), device=device), atol=1e-2):
        raise ValueError("Input n is not perpendicular to the plane")

    add_dim = False
    if points.ndim == 1:
        points = points.unsqueeze(0)
        add_dim = True

    assert points.shape[1] == 3
    # Every point p can be written as p = ctr + a e0 + b e1 + c n

    # If return_proj is True, we need to solve for (a, b)
    p_proj = None
    if return_proj:
        # solving for (a, b)
        A = torch.tensor([[1.0, e0.dot(e1)], [e0.dot(e1), 1.0]], dtype=torch.float32, device=device)
        B = torch.zeros((2, points.shape[0]), dtype=torch.float32, device=device)
        B[0, :] = torch.sum((points - plane_ctr.view(1, 3)) * e0.view(1, 3), dim=-1)
        B[1, :] = torch.sum((points - plane_ctr.view(1, 3)) * e1.view(1, 3), dim=-1)
        ab = A.inverse() @ B  # (2, P)
        p_proj = plane_ctr.view(1, 3) + ab.transpose(0, 1) @ torch.stack((e0, e1), dim=0)

    # solving for c
    # c = (point - ctr - a * e0 - b * e1).dot(n)
    direc = torch.sum((points - plane_ctr.view(1, 3)) * n.view(1, 3), dim=-1)
    ins = direc >= 0.0

    if add_dim:
        assert p_proj.shape[0] == 1
        p_proj = p_proj[0]

    return ins, p_proj
