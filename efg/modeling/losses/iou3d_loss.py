"""
Utilities for oriented bounding box manipulation and GIoU.
"""
import torch


def box_center_to_corners(b):
    """
    Converts a set of oriented bounding boxes from
    centered representation (x_c, y_c, w, h, theta) to corner representation (x0, y0, ..., x3, y3).
    Arguments:
        b (Tensor[N, 6]): boxes to be converted. They are
            expected to be in (x_c, y_c, w, h, c, s) format.
            * c, s: unnormalized cos, sin
    Returns:
        c (Tensor[N, 8]): converted boxes in (x0, y0, ..., x3, y3) format, where
            the corners are sorted counterclockwise.
    """
    x_c, y_c, w, h, c, s = b.unbind(-1)  # [N,]
    center = torch.stack([x_c, y_c], dim=-1).repeat(1, 4)  # [N, 8]

    dx = 0.5 * w
    dy = 0.5 * h
    c = c + 1e-5
    s = s + 1e-5
    cos = c / ((c**2 + s**2).sqrt() + 1e-10)
    sin = s / ((c**2 + s**2).sqrt() + 1e-10)

    dxcos = dx * cos
    dxsin = dx * sin
    dycos = dy * cos
    dysin = dy * sin

    dxy = [
        -dxcos + dysin,
        -dxsin - dycos,
        dxcos + dysin,
        dxsin - dycos,
        dxcos - dysin,
        dxsin + dycos,
        -dxcos - dysin,
        -dxsin + dycos,
    ]

    return center + torch.stack(dxy, dim=-1)  # [N, 8]


def box_corners_to_center(corners):
    """
    Arguments:
        corners (Tensor[N, 8]): boxes to be converted. They are
            expected to be in (x0, y0, ..., x3, y3) format, where the corners are sorted counterclockwise.
    Returns:
        b (Tensor[N, 6]): converted boxes in centered
            (x_c, y_c, w, h, c, s) format.
            * c, s: sin, cos before sigmoid
    """

    x0, y0, x1, y1, x2, y2, x3, y3 = corners.unbind(-1)

    x_c = (x0 + x2) / 2
    y_c = (y0 + y2) / 2

    wsin, wcos, hsin, hcos = (y1 - y0, x1 - x0, x0 + x1, y2 + y3)
    theta = torch.atan2(wsin, wcos)
    c = torch.cos(theta)
    s = torch.sin(theta)

    b = [x_c, y_c, (wsin**2 + wcos**2).sqrt(), (hsin**2 + hcos**2).sqrt(), c, s]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    x0, y0, x1, y1, x2, y2, x3, y3 = boxes.unbind(-1)  # [N,]
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 * ((x3 - x0) ** 2 + (y3 - y0) ** 2) ** 0.5  # [N,]


# looped
class Line:
    # ax + by + c = 0
    def __init__(self, v1, v2):  # v1, v2: [2,]
        self.a = v2[1] - v1[1]  # scalar
        self.b = v1[0] - v2[0]
        self.c = v2[0] * v1[1] - v2[1] * v1[0]

    def __call__(self, p):
        return self.a * p[0] + self.b * p[1] + self.c

    def intersection(self, other):
        # See e.g. https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        if not isinstance(other, Line):
            return NotImplementedError
        w = self.a * other.b - self.b * other.a
        return torch.stack(
            [(self.b * other.c - self.c * other.b) / w, (self.c * other.a - self.a * other.c) / w], dim=-1
        )  # [2,]


def box_inter(box1, box2):
    """
    Finds intersection convex polygon using sequential cut, then computes its area by counterclockwise cross product.
    https://stackoverflow.com/questions/44797713/calculate-the-area-of-intersection-of-two-rotated-rectangles-in-python/45268241
       Arguments:
           box1, box2 (Tensor[8], Tensor[8]): boxes to compute area of intersection. They are
               expected to be in (x0, y0, ..., x3, y3) format, where the corners are sorted counterclockwise.
       Returns:
           inter: torch.float32
    """
    intersection = box1.reshape([4, 2]).unbind(-2)  # [2,]
    box2_corners = box2.reshape([4, 2]).unbind(-2)  # [2,]

    for p, q in zip(box2_corners, box2_corners[1:] + box2_corners[:1]):
        new_intersection = list()

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        line_values = [line(t) for t in intersection]

        for s, t, s_value, t_value in zip(
            intersection, intersection[1:] + intersection[:1], line_values, line_values[1:] + line_values[:1]
        ):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return torch.tensor(0, dtype=torch.float).to(box1.device)

    return 0.5 * sum(p[0] * q[1] - p[1] * q[0] for p, q in zip(intersection, intersection[1:] + intersection[:1]))


def box_convex_hull(box1, box2):
    """
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain#Python
       Arguments:
           box1, box2 (Tensor[8], Tensor[8]): boxes to compute convex hull area. They are
               expected to be in (x0,y0, ..., x3,y3) format, where the corners are sorted counterclockwise.
       Returns:
           area: torch.float32
    """
    # Sort the points lexicographically (tuples are compared lexicographically).
    box1 = box1.reshape(4, 2).unbind(-2)
    box2 = box2.reshape(4, 2).unbind(-2)
    points = sorted(box1 + box2, key=lambda x: [x[0], x[1]])

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    convex_hull = lower[:-1] + upper[:-1]

    return 0.5 * sum(p[0] * q[1] - p[1] * q[0] for p, q in zip(convex_hull, convex_hull[1:] + convex_hull[:1]))


# vectorized
class Lines:
    # N instances of (ax + by + c = 0)
    def __init__(self, v1, v2):  # v1, v2: [N, 2]
        self.a = v2[:, 1] - v1[:, 1]  # [N,]
        self.b = v1[:, 0] - v2[:, 0]
        self.c = v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]
        self.eps = 1e-10

    def __call__(self, p):  # [N, 2]
        return self.a * p[:, 0] + self.b * p[:, 1] + self.c  # [N,]

    def intersection(self, others):
        if not isinstance(others, Lines):
            return NotImplementedError
        w = self.a * others.b - self.b * others.a + self.eps
        inter = torch.stack(
            [(self.b * others.c - self.c * others.b) / w, (self.c * others.a - self.a * others.c) / w], dim=-1
        )
        return inter  # [N, 2]


def cuts(polygons, sizes, p, q):
    """
    vectorized polygon cut
    Arguments:
        polygons (Tensor[N, K, 2])
        sizes (Tensor[N,])
        p (Tensor[N, 2])
        q (Tensor[N, 2])
    Returns:
        new_polygons (Tensor[N, K+1, 2])
        new_sizes (Tensor[N,])
    new_polygons = list()
    """
    N = polygons.shape[0]
    K = polygons.shape[1]

    # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
    # any point p with line(p) > 0 is on the "outside".
    lines = Lines(p, q)

    polygons_abc = polygons  # [N, K, 2]
    polygons_bca = polygons.clone()  # [:, [1, 2, 3, 0], :]
    polygons_bca[:, :-1, :] = polygons_abc[:, 1:, :]
    polygons_bca[torch.arange(N), sizes - 1, :] = polygons_abc[:, 0, :]

    v_abc = torch.stack([lines(polygons_abc[:, k, :]) for k in range(K)], dim=1)  # [N, K]
    v_bca = torch.stack([lines(polygons_bca[:, k, :]) for k in range(K)], dim=1)  # [N, K]

    # use new_polygons as stack and new_sizes as stack pointer
    # iterate and push new points
    new_polygons = torch.zeros(N, K + 1, 2).to(polygons.device).fill_(1e5)  # [N, K + 1, 2]
    new_sizes = torch.zeros(N).to(polygons.device).long()  # [N,]
    for k in range(K):
        s = polygons_abc[:, k, :].clone()  # [N, 2]
        t = polygons_bca[:, k, :].clone()
        s_v = v_abc[:, k].clone()  # [N,]
        t_v = v_bca[:, k].clone()

        # only keep valid points
        valid = sizes > k  # [N,]
        s[~valid, :] = 0
        t[~valid, :] = 0
        s_v[~valid] = 0
        t_v[~valid] = 0

        # push preserved points to stack
        mask = (s_v <= 0) & valid
        push = s.clone()  # [N, 2]
        keep = new_polygons[torch.arange(N), new_sizes - 1, :]  # [N, 2]

        push[~mask, :] = 0
        keep[mask, :] = 0

        new_sizes = new_sizes + mask.long().squeeze(-1)
        new_polygons[torch.arange(N), new_sizes - 1, :] = push + keep

        # push intersection points to stack
        mask = (s_v * t_v < 0) & valid
        push = lines.intersection(Lines(s, t))  # [N, 2]
        keep = new_polygons[torch.arange(N), new_sizes - 1, :]  # [N, 2]

        push[~mask, :] = 0
        keep[mask, :] = 0

        new_sizes = new_sizes + mask.long().squeeze(-1)
        new_polygons[torch.arange(N), new_sizes - 1, :] = push + keep

    return new_polygons, new_sizes


def box_inter_tensor(boxes1, boxes2):
    """
    Finds intersection convex polygon using sequential cut, then computes its area by counterclockwise cross product.
    https://stackoverflow.com/questions/44797713/calculate-the-area-of-intersection-of-two-rotated-rectangles-in-python/45268241
       Arguments:
           boxes1, boxes2 (Tensor[N, 8], Tensor[M, 8]): boxes to compute area of intersection. They are
               expected to be in (x0, y0, ..., x3, y3) format, where the corners are sorted counterclockwise.
       Returns:
           inter (Tensor[N, M]) pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    N = boxes1.shape[-2]
    M = boxes2.shape[-2]

    boxes1 = boxes1.reshape([-1, 4, 2])
    boxes2 = boxes2.reshape([-1, 4, 2])

    # vectorized intersection computation
    inter_xy = boxes1.unsqueeze(1).expand(-1, M, -1, -1).reshape([N * M, 4, 2])  # [N * M, 4, 2]
    cut_rect = boxes2.unsqueeze(0).expand(N, -1, -1, -1).reshape([N * M, 4, 2])  # [N * M, 4, 2]
    sizes = (
        torch.zeros(
            [
                N * M,
            ]
        )
        .to(boxes1.device)
        .long()
        .fill_(4)
    )  # [N, M]

    inter_xy, sizes = cuts(inter_xy, sizes, cut_rect[:, 0, :], cut_rect[:, 1, :])  # [N * M, 5, 2]
    inter_xy, sizes = cuts(inter_xy, sizes, cut_rect[:, 1, :], cut_rect[:, 2, :])  # [N * M, 6, 2]
    inter_xy, sizes = cuts(inter_xy, sizes, cut_rect[:, 2, :], cut_rect[:, 3, :])  # [N * M, 7, 2]
    inter_xy, sizes = cuts(inter_xy, sizes, cut_rect[:, 3, :], cut_rect[:, 0, :])  # [N * M, 8, 2]

    # compute area
    inter_abc = inter_xy  # [N*M, 8, 2]
    inter_bca = inter_abc.clone()
    inter_bca[:, :-1, :] = inter_abc[:, 1:, :]
    inter_bca[torch.arange(N * M), sizes - 1, :] = inter_abc[:, 0, :]

    inter = inter_abc[:, :, 0] * inter_bca[:, :, 1] - inter_abc[:, :, 1] * inter_bca[:, :, 0]

    sizes = sizes.unsqueeze(-1).expand([-1, 8])  # [N * M, 8]
    inter[sizes <= 2] = 0
    inter[sizes <= torch.arange(8).unsqueeze(0).to(boxes1.device)] = 0

    return 0.5 * inter.reshape([N, M, -1]).sum(dim=-1)  # [N, M]


def box_inter_tensor_diag(boxes1, boxes2):
    """
    Finds intersection convex polygon using sequential cut, then computes its area by counterclockwise cross product.
    https://stackoverflow.com/questions/44797713/calculate-the-area-of-intersection-of-two-rotated-rectangles-in-python/45268241
       Arguments:
           boxes1, boxes2 (Tensor[N, 8], Tensor[N, 8]): boxes to compute area of intersection. They are
               expected to be in (x0, y0, ..., x3, y3) format, where the corners are sorted counterclockwise.
       Returns:
           inter (Tensor[N]) pairwise matrix, where N = len(boxes1) and N = len(boxes2)
    """
    assert boxes1.shape[0] == boxes2.shape[0]

    N = boxes1.shape[0]

    boxes1 = boxes1.reshape([-1, 4, 2])
    boxes2 = boxes2.reshape([-1, 4, 2])

    # vectorized intersection computation
    inter_xy = boxes1
    cut_rect = boxes2
    sizes = (
        torch.zeros(
            [
                N,
            ],
            device=boxes1.device,
        )
        .long()
        .fill_(4)
    )

    # inter_xy = boxes1.unsqueeze(1).expand(-1, M, -1, -1).reshape([N * M, 4, 2])  # [N * M, 4, 2]
    # cut_rect = boxes2.unsqueeze(0).expand(N, -1, -1, -1).reshape([N * M, 4, 2])  # [N * M, 4, 2]
    # sizes = torch.zeros([N * M,]).to(boxes1.device).long().fill_(4)  # [N, M]

    inter_xy, sizes = cuts(inter_xy, sizes, cut_rect[:, 0, :], cut_rect[:, 1, :])  # [N * M, 5, 2]
    inter_xy, sizes = cuts(inter_xy, sizes, cut_rect[:, 1, :], cut_rect[:, 2, :])  # [N * M, 6, 2]
    inter_xy, sizes = cuts(inter_xy, sizes, cut_rect[:, 2, :], cut_rect[:, 3, :])  # [N * M, 7, 2]
    inter_xy, sizes = cuts(inter_xy, sizes, cut_rect[:, 3, :], cut_rect[:, 0, :])  # [N * M, 8, 2]

    # compute area
    inter_abc = inter_xy  # [N*M, 8, 2]
    inter_bca = inter_abc.clone()
    inter_bca[:, :-1, :] = inter_abc[:, 1:, :]
    inter_bca[torch.arange(N), sizes - 1, :] = inter_abc[:, 0, :]

    inter = inter_abc[:, :, 0] * inter_bca[:, :, 1] - inter_abc[:, :, 1] * inter_bca[:, :, 0]

    sizes = sizes.unsqueeze(-1).expand([-1, 8])  # [N * M, 8]
    inter[sizes <= 2] = 0
    inter[sizes <= torch.arange(8).unsqueeze(0).to(boxes1.device)] = 0

    return 0.5 * inter.reshape([N, -1]).sum(dim=-1)  # [N, M]


def box_convex_hull_tensor(boxes1, boxes2):
    """
    Arguments:
        boxes1, boxes2 (Tensor[N, 8], Tensor[M, 8]): boxes to compute convex hull area. They are
            expected to be in (x0, y0, ..., x3, y3) format, where the corners are sorted counterclockwise.
    Returns:
        hull (Tensor[N, M]) pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """

    N = boxes1.shape[-2]
    M = boxes2.shape[-2]

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):  # [N, 2]
        return (a[:, 0] - o[:, 0]) * (b[:, 1] - o[:, 1]) - (a[:, 1] - o[:, 1]) * (b[:, 0] - o[:, 0])

    # concat and sort
    # trick: add neligiable noise to enforce unique x-values
    boxes1 = boxes1.reshape([-1, 4, 2]).unsqueeze(1).expand(-1, M, -1, -1).reshape([N * M, 4, 2])  # [N * M, 4, 2]
    boxes2 = boxes2.reshape([-1, 4, 2]).unsqueeze(0).expand(N, -1, -1, -1).reshape([N * M, 4, 2])  # [N * M, 4, 2]
    boxes = torch.cat([boxes1, boxes2], dim=1)  # [N * M, 8, 2]
    boxes = boxes + 1e-5 * torch.randn([N * M, 8, 2]).to(boxes1.device)
    _, indices = boxes.sort(dim=1, descending=False)  # [N * M, 8, 2]
    indices = indices[:, :, 0].unsqueeze(-1).expand([-1, -1, 2])  # [N * M, 8, 2]
    boxes = boxes.gather(dim=1, index=indices)

    # build lower / upper hull
    lower = torch.zeros(N * M, 8, 2).to(boxes.device).fill_(1e10)
    lower_sizes = torch.zeros(N * M).to(boxes.device).long()  # stack top
    upper = torch.zeros(N * M, 8, 2).to(boxes.device).fill_(1e10)
    upper_sizes = torch.zeros(N * M).to(boxes.device).long()  # stack top

    for k in range(8):
        while True:
            mask = (lower_sizes >= 2) & (
                cross(
                    lower[torch.arange(N * M), lower_sizes - 2, :],
                    lower[torch.arange(N * M), lower_sizes - 1, :],
                    boxes[:, k, :],
                )
                <= 0
            )
            lower_sizes = lower_sizes - mask.long()
            if mask.any():
                continue
            break
        lower_sizes = lower_sizes + 1
        lower[torch.arange(N * M), lower_sizes - 1, :] = boxes[:, k, :]

        while True:
            mask = (upper_sizes >= 2) & (
                cross(
                    upper[torch.arange(N * M), upper_sizes - 2, :],
                    upper[torch.arange(N * M), upper_sizes - 1, :],
                    boxes[:, 7 - k, :],
                )
                <= 0
            )
            upper_sizes = upper_sizes - mask.long()
            if mask.any():
                continue
            break
        upper_sizes = upper_sizes + 1
        upper[torch.arange(N * M), upper_sizes - 1, :] = boxes[:, 7 - k, :]

    # concatenation of the lower and upper hulls gives the convex hull.
    # last point of each list is omitted because it is repeated at the beginning of the other list.

    convex_hull = torch.zeros(N * M, 8, 2).to(boxes.device).fill_(1e10)
    sizes = torch.zeros(N * M).to(boxes.device).long()

    for k in range(8):
        mask = lower_sizes > k + 1
        convex_hull[mask, k, :] = lower[mask, k, :]
        sizes = sizes + mask.long()

    for k in range(8):
        mask = upper_sizes > k + 1

        push = upper[:, k, :].clone()
        keep = convex_hull[torch.arange(N * M), sizes - 1, :]

        push[~mask, :] = 0
        keep[mask, :] = 0

        sizes = sizes + mask.long()
        convex_hull[torch.arange(N * M), sizes - 1, :] = push + keep

    # compute area
    hull_abc = convex_hull  # [N * M, 8, 2]
    hull_bca = hull_abc.clone()
    hull_bca[:, :-1, :] = hull_abc[:, 1:, :]
    hull_bca[torch.arange(N * M), sizes - 1, :] = hull_abc[:, 0, :]

    hull = hull_abc[:, :, 0] * hull_bca[:, :, 1] - hull_abc[:, :, 1] * hull_bca[:, :, 0]

    sizes = sizes.unsqueeze(-1).expand([-1, 8])  # [N * M, 8]
    hull[sizes <= 2] = 0
    hull[sizes <= torch.arange(8).unsqueeze(0).to(boxes1.device)] = 0

    return 0.5 * hull.reshape([N, M, -1]).sum(dim=-1)  # [N, M]


def box_convex_hull_tensor_diag(boxes1, boxes2):
    """
    Arguments:
        boxes1, boxes2 (Tensor[N, 8], Tensor[N, 8]): boxes to compute convex hull area. They are
            expected to be in (x0, y0, ..., x3, y3) format, where the corners are sorted counterclockwise.
    Returns:
        hull (Tensor[N, M]) pairwise matrix, where N = len(boxes1) and N = len(boxes2)
    """

    assert boxes1.shape[0] == boxes2.shape[0]

    N = boxes1.shape[0]

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):  # [N, 2]
        return (a[:, 0] - o[:, 0]) * (b[:, 1] - o[:, 1]) - (a[:, 1] - o[:, 1]) * (b[:, 0] - o[:, 0])

    # concat and sort
    # trick: add neligiable noise to enforce unique x-values
    boxes1 = boxes1.reshape([-1, 4, 2])  # .unsqueeze(1).expand(-1, M, -1, -1).reshape([N * M, 4, 2])  # [N * M, 4, 2]
    boxes2 = boxes2.reshape([-1, 4, 2])  # .unsqueeze(0).expand(N, -1, -1, -1).reshape([N * M, 4, 2])  # [N * M, 4, 2]
    boxes = torch.cat([boxes1, boxes2], dim=1)  # [N * M, 8, 2]
    boxes = boxes + 1e-5 * torch.randn([N, 8, 2]).to(boxes1.device)
    _, indices = boxes.sort(dim=1, descending=False)  # [N * M, 8, 2]
    indices = indices[:, :, 0].unsqueeze(-1).expand([-1, -1, 2])  # [N * M, 8, 2]
    boxes = boxes.gather(dim=1, index=indices)

    # build lower / upper hull
    lower = torch.zeros(N, 8, 2).to(boxes.device).fill_(1e10)
    lower_sizes = torch.zeros(N).to(boxes.device).long()  # stack top
    upper = torch.zeros(N, 8, 2).to(boxes.device).fill_(1e10)
    upper_sizes = torch.zeros(N).to(boxes.device).long()  # stack top

    for k in range(8):
        while True:
            mask = (lower_sizes >= 2) & (
                cross(
                    lower[torch.arange(N), lower_sizes - 2, :],
                    lower[torch.arange(N), lower_sizes - 1, :],
                    boxes[:, k, :],
                )
                <= 0
            )
            lower_sizes = lower_sizes - mask.long()
            if mask.any():
                continue
            break
        lower_sizes = lower_sizes + 1
        lower[torch.arange(N), lower_sizes - 1, :] = boxes[:, k, :]

        while True:
            mask = (upper_sizes >= 2) & (
                cross(
                    upper[torch.arange(N), upper_sizes - 2, :],
                    upper[torch.arange(N), upper_sizes - 1, :],
                    boxes[:, 7 - k, :],
                )
                <= 0
            )
            upper_sizes = upper_sizes - mask.long()
            if mask.any():
                continue
            break
        upper_sizes = upper_sizes + 1
        upper[torch.arange(N), upper_sizes - 1, :] = boxes[:, 7 - k, :]

    # concatenation of the lower and upper hulls gives the convex hull.
    # last point of each list is omitted because it is repeated at the beginning of the other list.

    convex_hull = torch.zeros(N, 8, 2).to(boxes.device).fill_(1e10)
    sizes = torch.zeros(N).to(boxes.device).long()

    for k in range(8):
        mask = lower_sizes > k + 1
        convex_hull[mask, k, :] = lower[mask, k, :]
        sizes = sizes + mask.long()

    for k in range(8):
        mask = upper_sizes > k + 1

        push = upper[:, k, :].clone()
        keep = convex_hull[torch.arange(N), sizes - 1, :]

        push[~mask, :] = 0
        keep[mask, :] = 0

        sizes = sizes + mask.long()
        convex_hull[torch.arange(N), sizes - 1, :] = push + keep

    # compute area
    hull_abc = convex_hull  # [N * M, 8, 2]
    hull_bca = hull_abc.clone()
    hull_bca[:, :-1, :] = hull_abc[:, 1:, :]
    hull_bca[torch.arange(N), sizes - 1, :] = hull_abc[:, 0, :]

    hull = hull_abc[:, :, 0] * hull_bca[:, :, 1] - hull_abc[:, :, 1] * hull_bca[:, :, 0]

    sizes = sizes.unsqueeze(-1).expand([-1, 8])  # [N * M, 8]
    hull[sizes <= 2] = 0
    hull[sizes <= torch.arange(8).unsqueeze(0).to(boxes1.device)] = 0

    return 0.5 * hull.reshape([N, -1]).sum(dim=-1)  # [N, M]


def box_iou(boxes1, boxes2):
    """
    Arguments:
        boxes1, boxes2 (Tensor[N, 8], Tensor[M, 8]): boxes to compute IoU. They are
            expected to be in (x0, y0, ..., x3, y3) format, where the corners are sorted counterclockwise.
    Returns:
        iou: [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
        union: [N, M] pairwise matrix
    """
    area1 = box_area(boxes1)  # [N,]
    area2 = box_area(boxes2)  # [M,]

    """
    inter = torch.zeros([boxes1.shape[-2], boxes2.shape[-2]]).to(boxes1.device)  # [N, M]
    inter = torch.zeros_like(boxes1)
    for n in range(boxes1.shape[-2]):
        for m in range(boxes2.shape[-2]):
            inter[n, m] = box_inter(boxes1[n, :], boxes2[m, :])
    """

    inter = box_inter_tensor(boxes1, boxes2)
    union = area1[..., None] + area2[None] - inter  # [N, M]

    iou = inter / union  # [N, M]
    return iou, union


def box_iou_diag(boxes1, boxes2):
    """
    Arguments:
        boxes1, boxes2 (Tensor[N, 8], Tensor[M, 8]): boxes to compute IoU. They are
            expected to be in (x0, y0, ..., x3, y3) format, where the corners are sorted counterclockwise.
    Returns:
        iou: [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
        union: [N, M] pairwise matrix
    """
    area1 = box_area(boxes1)  # [N,]
    area2 = box_area(boxes2)  # [M,]

    """
    inter = torch.zeros([boxes1.shape[-2], boxes2.shape[-2]]).to(boxes1.device)  # [N, M]
    inter = torch.zeros_like(boxes1)
    for n in range(boxes1.shape[-2]):
        for m in range(boxes2.shape[-2]):
            inter[n, m] = box_inter(boxes1[n, :], boxes2[m, :])
    """

    inter = box_inter_tensor_diag(boxes1, boxes2)
    union = area1 + area2 - inter  # [N, M]

    iou = inter / union  # [N, M]
    return iou, union


def generalized_box_iou3d(boxes1, boxes2):
    """
    boxes1: N x 8 (x, y, z, l, w, h, cos(yaw), sin(yaw))
    boxes2: M x 8
    First convert xyzlwhyaw to bev box to corners
    The boxes should be in corners [x0,y0, ... x3,y3] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """

    corners1 = box_center_to_corners(boxes1[..., [0, 1, 3, 4, 6, 7]])
    corners2 = box_center_to_corners(boxes2[..., [0, 1, 3, 4, 6, 7]])
    iou, union = box_iou_diag(corners1, corners2)
    inter = iou * union

    zmax1 = boxes1[..., 2] + boxes1[..., 5] * 0.5
    zmin1 = boxes1[..., 2] - boxes1[..., 5] * 0.5
    zmax2 = boxes2[..., 2] + boxes2[..., 5] * 0.5
    zmin2 = boxes2[..., 2] - boxes2[..., 5] * 0.5

    z_overlap = (torch.min(zmax1, zmax2) - torch.max(zmin1, zmin2)).clamp_min(0.0)
    inter3d = inter * z_overlap

    vol1 = boxes1[..., 3:6].prod(dim=-1)
    vol2 = boxes2[..., 3:6].prod(dim=-1)
    union3d = vol1 + vol2 - inter3d

    iou3d = inter3d / union3d

    hull = box_convex_hull_tensor_diag(corners1, corners2)
    z_range = (torch.max(zmax1, zmax2) - torch.min(zmin1, zmin2)).clamp_min(0.0)
    hull3d = hull * z_range

    return iou3d - (hull3d - union3d) / hull3d


def generalized_box_iou_bev(boxes1, boxes2):
    """
    boxes1: N x 8 (x, y, z, l, w, h, cos(yaw), sin(yaw))
    boxes2: M x 8
    First convert xyzlwhyaw to bev box to corners
    The boxes should be in corners [x0,y0, ... x3,y3] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """

    corners1 = box_center_to_corners(boxes1[..., [0, 1, 3, 4, 6, 7]])
    corners2 = box_center_to_corners(boxes2[..., [0, 1, 3, 4, 6, 7]])
    iou, union = box_iou(corners1, corners2)

    hull = box_convex_hull_tensor(corners1, corners2)

    return iou - (hull - union) / hull
