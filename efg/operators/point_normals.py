from typing import Optional, Tuple, Union

import torch

from efg.geometry.symeig3x3 import symeig3x3
from efg.operators.knn import knn_points


def masked_gather(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Helper function for torch.gather to collect the points at
    the given indices in idx where some of the indices might be -1 to
    indicate padding. These indices are first replaced with 0.
    Then the points are gathered after which the padded values
    are set to 0.0.

    Args:
        points: (N, P, D) float32 tensor of points
        idx: (N, K) or (N, P, K) long tensor of indices into points, where
            some indices are -1 to indicate padding

    Returns:
        selected_points: (N, K, D) float32 tensor of points
            at the given indices
    """

    if len(idx) != len(points):
        raise ValueError("points and idx must have the same batch dimension")

    N, P, D = points.shape

    if idx.ndim == 3:
        # Case: KNN, Ball Query where idx is of shape (N, P', K)
        # where P' is not necessarily the same as P as the
        # points may be gathered from a different pointcloud.
        K = idx.shape[2]
        # Match dimensions for points and indices
        idx_expanded = idx[..., None].expand(-1, -1, -1, D)
        points = points[:, :, None, :].expand(-1, -1, K, -1)
    elif idx.ndim == 2:
        # Farthest point sampling where idx is of shape (N, K)
        idx_expanded = idx[..., None].expand(-1, -1, D)
    else:
        raise ValueError("idx format is not supported %s" % repr(idx.shape))

    idx_expanded_mask = idx_expanded.eq(-1)
    idx_expanded = idx_expanded.clone()
    # Replace -1 values with 0 for gather
    idx_expanded[idx_expanded_mask] = 0
    # Gather points
    selected_points = points.gather(dim=1, index=idx_expanded)
    # Replace padded values
    selected_points[idx_expanded_mask] = 0.0
    return selected_points


def wmean(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    dim: Union[int, Tuple[int]] = -2,
    keepdim: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Finds the mean of the input tensor across the specified dimension.
    If the `weight` argument is provided, computes weighted mean.
    Args:
        x: tensor of shape `(*, D)`, where D is assumed to be spatial;
        weights: if given, non-negative tensor of shape `(*,)`. It must be
            broadcastable to `x.shape[:-1]`. Note that the weights for
            the last (spatial) dimension are assumed same;
        dim: dimension(s) in `x` to average over;
        keepdim: tells whether to keep the resulting singleton dimension.
        eps: minimum clamping value in the denominator.
    Returns:
        the mean tensor:
        * if `weights` is None => `mean(x, dim)`,
        * otherwise => `sum(x*w, dim) / max{sum(w, dim), eps}`.
    """
    args = {"dim": dim, "keepdim": keepdim}

    if weight is None:
        # pyre-fixme[6]: For 1st param expected `Optional[dtype]` but got
        #  `Union[Tuple[int], int]`.
        return x.mean(**args)

    if any(
        xd != wd and xd != 1 and wd != 1
        for xd, wd in zip(x.shape[-2::-1], weight.shape[::-1])
    ):
        raise ValueError("wmean: weights are not compatible with the tensor")

    # pyre-fixme[6]: For 1st param expected `Optional[dtype]` but got
    #  `Union[Tuple[int], int]`.
    return (x * weight[..., None]).sum(**args) / weight[..., None].sum(**args).clamp(
        eps
    )


def eyes(
    dim: int,
    N: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generates a batch of `N` identity matrices of shape `(N, dim, dim)`.

    Args:
        **dim**: The dimensionality of the identity matrices.
        **N**: The number of identity matrices.
        **device**: The device to be used for allocating the matrices.
        **dtype**: The datatype of the matrices.

    Returns:
        **identities**: A batch of identity matrices of shape `(N, dim, dim)`.
    """
    identities = torch.eye(dim, device=device, dtype=dtype)
    return identities[None].repeat(N, 1, 1)


def convert_pointclouds_to_tensor(pcl: Union[torch.Tensor, ]):
    """
    If `type(pcl)==Pointclouds`, converts a `pcl` object to a
    padded representation and returns it together with the number of points
    per batch. Otherwise, returns the input itself with the number of points
    set to the size of the second dimension of `pcl`.
    """
    if is_pointclouds(pcl):
        X = pcl.points_padded()  # type: ignore
        num_points = pcl.num_points_per_cloud()  # type: ignore
    elif torch.is_tensor(pcl):
        X = pcl
        num_points = X.shape[1] * torch.ones(  # type: ignore
            # pyre-fixme[16]: Item `Pointclouds` of `Union[Pointclouds, Tensor]` has
            #  no attribute `shape`.
            X.shape[0],
            device=X.device,
            dtype=torch.int64,
        )
    else:
        raise ValueError(
            "The inputs X, Y should be either Pointclouds objects or tensors."
        )
    return X, num_points


def is_pointclouds(pcl: Union[torch.Tensor, ]) -> bool:
    """Checks whether the input `pcl` is an instance of `Pointclouds`
    by checking the existence of `points_padded` and `num_points_per_cloud`
    functions.
    """
    return hasattr(pcl, "points_padded") and hasattr(pcl, "num_points_per_cloud")


def get_point_covariances(
    points_padded: torch.Tensor,
    num_points_per_cloud: torch.Tensor,
    neighborhood_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the per-point covariance matrices by of the 3D locations of
    K-nearest neighbors of each point.

    Args:
        **points_padded**: Input point clouds as a padded tensor
            of shape `(minibatch, num_points, dim)`.
        **num_points_per_cloud**: Number of points per cloud
            of shape `(minibatch,)`.
        **neighborhood_size**: Number of nearest neighbors for each point
            used to estimate the covariance matrices.

    Returns:
        **covariances**: A batch of per-point covariance matrices
            of shape `(minibatch, dim, dim)`.
        **k_nearest_neighbors**: A batch of `neighborhood_size` nearest
            neighbors for each of the point cloud points
            of shape `(minibatch, num_points, neighborhood_size, dim)`.
    """
    # get K nearest neighbor idx for each point in the point cloud
    k_nearest_neighbors = knn_points(
        points_padded,
        points_padded,
        lengths1=num_points_per_cloud,
        lengths2=num_points_per_cloud,
        K=neighborhood_size,
        return_nn=True,
    ).knn
    # obtain the mean of the neighborhood
    pt_mean = k_nearest_neighbors.mean(2, keepdim=True)
    # compute the diff of the neighborhood and the mean of the neighborhood
    central_diff = k_nearest_neighbors - pt_mean
    # per-nn-point covariances
    per_pt_cov = central_diff.unsqueeze(4) * central_diff.unsqueeze(3)
    # per-point covariances
    covariances = per_pt_cov.mean(2)

    return covariances, k_nearest_neighbors


def estimate_pointcloud_normals(
    pointclouds: Union[torch.Tensor, ],
    neighborhood_size: int = 50,
    disambiguate_directions: bool = True,
    *,
    use_symeig_workaround: bool = True,
) -> torch.Tensor:
    """
    Estimates the normals of a batch of `pointclouds`.

    The function uses `estimate_pointcloud_local_coord_frames` to estimate
    the normals. Please refer to that function for more detailed information.

    Args:
      **pointclouds**: Batch of 3-dimensional points of shape
        `(minibatch, num_point, 3)` or a `Pointclouds` object.
      **neighborhood_size**: The size of the neighborhood used to estimate the
        geometry around each point.
      **disambiguate_directions**: If `True`, uses the algorithm from [1] to
        ensure sign consistency of the normals of neighboring points.
      **use_symeig_workaround**: If `True`, uses a custom eigenvalue
        calculation.

    Returns:
      **normals**: A tensor of normals for each input point
        of shape `(minibatch, num_point, 3)`.
        If `pointclouds` are of `Pointclouds` class, returns a padded tensor.

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """

    curvatures, local_coord_frames = estimate_pointcloud_local_coord_frames(
        pointclouds,
        neighborhood_size=neighborhood_size,
        disambiguate_directions=disambiguate_directions,
        use_symeig_workaround=use_symeig_workaround,
    )

    # the normals correspond to the first vector of each local coord frame
    normals = local_coord_frames[:, :, :, 0]

    return normals


def estimate_pointcloud_local_coord_frames(
    pointclouds: Union[torch.Tensor, ],
    neighborhood_size: int = 50,
    disambiguate_directions: bool = True,
    *,
    use_symeig_workaround: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimates the principal directions of curvature (which includes normals)
    of a batch of `pointclouds`.

    The algorithm first finds `neighborhood_size` nearest neighbors for each
    point of the point clouds, followed by obtaining principal vectors of
    covariance matrices of each of the point neighborhoods.
    The main principal vector corresponds to the normals, while the
    other 2 are the direction of the highest curvature and the 2nd highest
    curvature.

    Note that each principal direction is given up to a sign. Hence,
    the function implements `disambiguate_directions` switch that allows
    to ensure consistency of the sign of neighboring normals. The implementation
    follows the sign disabiguation from SHOT descriptors [1].

    The algorithm also returns the curvature values themselves.
    These are the eigenvalues of the estimated covariance matrices
    of each point neighborhood.

    Args:
      **pointclouds**: Batch of 3-dimensional points of shape
        `(minibatch, num_point, 3)` or a `Pointclouds` object.
      **neighborhood_size**: The size of the neighborhood used to estimate the
        geometry around each point.
      **disambiguate_directions**: If `True`, uses the algorithm from [1] to
        ensure sign consistency of the normals of neighboring points.
      **use_symeig_workaround**: If `True`, uses a custom eigenvalue
        calculation.

    Returns:
      **curvatures**: The three principal curvatures of each point
        of shape `(minibatch, num_point, 3)`.
        If `pointclouds` are of `Pointclouds` class, returns a padded tensor.
      **local_coord_frames**: The three principal directions of the curvature
        around each point of shape `(minibatch, num_point, 3, 3)`.
        The principal directions are stored in columns of the output.
        E.g. `local_coord_frames[i, j, :, 0]` is the normal of
        `j`-th point in the `i`-th pointcloud.
        If `pointclouds` are of `Pointclouds` class, returns a padded tensor.

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """

    points_padded, num_points = convert_pointclouds_to_tensor(pointclouds)

    ba, N, dim = points_padded.shape
    if dim != 3:
        raise ValueError(
            "The pointclouds argument has to be of shape (minibatch, N, 3)"
        )

    if (num_points <= neighborhood_size).any():
        raise ValueError(
            "The neighborhood_size argument has to be"
            + " >= size of each of the point clouds."
        )

    # undo global mean for stability
    # TODO: replace with tutil.wmean once landed
    pcl_mean = points_padded.sum(1) / num_points[:, None]
    points_centered = points_padded - pcl_mean[:, None, :]

    # get the per-point covariance and nearest neighbors used to compute it
    cov, knns = get_point_covariances(points_centered, num_points, neighborhood_size)

    # get the local coord frames as principal directions of
    # the per-point covariance
    # this is done with torch.symeig / torch.linalg.eigh, which returns the
    # eigenvectors (=principal directions) in an ascending order of their
    # corresponding eigenvalues, and the smallest eigenvalue's eigenvector
    # corresponds to the normal direction; or with a custom equivalent.
    if use_symeig_workaround:
        curvatures, local_coord_frames = symeig3x3(cov, eigenvectors=True)
    else:
        curvatures, local_coord_frames = torch.linalg.eigh(cov)

    # disambiguate the directions of individual principal vectors
    if disambiguate_directions:
        # disambiguate normal
        n = _disambiguate_vector_directions(
            points_centered, knns, local_coord_frames[:, :, :, 0]
        )
        # disambiguate the main curvature
        z = _disambiguate_vector_directions(
            points_centered, knns, local_coord_frames[:, :, :, 2]
        )
        # the secondary curvature is just a cross between n and z
        y = torch.cross(n, z, dim=2)
        # cat to form the set of principal directions
        local_coord_frames = torch.stack((n, y, z), dim=3)

    return curvatures, local_coord_frames


def _disambiguate_vector_directions(pcl, knns, vecs: torch.Tensor) -> torch.Tensor:
    """
    Disambiguates normal directions according to [1].

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """
    # parse out K from the shape of knns
    K = knns.shape[2]
    # the difference between the mean of each neighborhood and
    # each element of the neighborhood
    df = knns - pcl[:, :, None]
    # projection of the difference on the principal direction
    proj = (vecs[:, :, None] * df).sum(3)
    # check how many projections are positive
    n_pos = (proj > 0).type_as(knns).sum(2, keepdim=True)
    # flip the principal directions where number of positive correlations
    flip = (n_pos < (0.5 * K)).type_as(knns)
    vecs = (1.0 - 2.0 * flip) * vecs
    return vecs
