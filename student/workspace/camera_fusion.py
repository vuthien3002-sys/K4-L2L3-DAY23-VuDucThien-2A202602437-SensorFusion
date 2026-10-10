"""Camera field-of-view checks and pinhole measurement modeling.

Part G supplies visibility, projection, and pixel covariance (docs/HUONG_DAN_KY_THUAT.md §2).
The platform differentiates projection using a chain-rule Jacobian.
"""

from __future__ import annotations

from typing import Any
from typing import Sequence

import numpy as np

from fusion_lab.workspace_support import get_tracking_params

Matrix = np.matrix | np.ndarray

# vi: from fusion_lab.workspace_support import get_tracking_params

MIN_DEPTH = 1e-6


def _sensor_position(x: Matrix, sensor: Any) -> np.ndarray:
    """Transform the state position into sensor coordinates: p_s = R p + t."""
    position = np.asarray(x, dtype=float).reshape(-1)[:3]
    transform = np.asarray(sensor.veh_to_sens, dtype=float)
    return transform[:3, :3] @ position + transform[:3, 3]


def _is_camera(sensor: Any) -> bool:
    return getattr(sensor, "name", None) == "camera"


def is_in_field_of_view(x: Matrix, sensor: Any) -> bool:
    """Return True if state x is visible within the sensor horizontal field of view.

    Args:
        x: State vector (6x1) with position in vehicle frame.
        sensor: Lidar or camera adapter with ``veh_to_sens`` and ``fov``
            (radians).

    Returns:
        True if sensor coordinates are finite and the horizontal angle is within
        ``sensor.fov``. A camera additionally requires depth > 1e-6.
    """
    # vi: TODO Part G — p_s = R @ p + t; loại tọa độ không hữu hạn.
    # vi: Camera cần x_s > 1e-6; FOV từ nội tại và bề rộng ảnh.
    # vi: Với cả lidar/camera: kiểm tra atan2(y_s, x_s) nằm trong sensor.fov.
    p_s = _sensor_position(x, sensor)
    if not np.isfinite(p_s).all():
        return False
    fov_min, fov_max = sensor.fov
    if _is_camera(sensor):
        if p_s[0] <= MIN_DEPTH:
            return False
        if hasattr(sensor, "image_width"):
            # Pixel columns [0, width] map to angles atan((c_i - u) / f_i).
            fov_min = np.arctan((sensor.c_i - sensor.image_width) / sensor.f_i)
            fov_max = np.arctan(sensor.c_i / sensor.f_i)
    angle = np.arctan2(p_s[1], p_s[0])
    return bool(fov_min <= angle <= fov_max)


def camera_measurement_prediction(x: Matrix, sensor: Any) -> Matrix:
    """Predict image-plane measurement h(x) using the pinhole camera model.

    Args:
        x: State vector.
        sensor: Camera with intrinsics ``f_i, f_j, c_i, c_j``.

    Returns:
        2x1 predicted pixel coordinates as ``np.matrix``.

    Raises:
        ValueError: With coordinate context if sensor coordinates are nonfinite
            or depth is at most 1e-6.
    """
    # vi: TODO Part G — tính p_s = R @ p + t; trước phép chia kiểm tra hữu hạn
    # vi: và x_s > 1e-6, ngược lại raise ValueError có tọa độ.
    # vi: u = c_i - f_i * y_s/x_s; v = c_j - f_j * z_s/x_s.
    p_s = _sensor_position(x, sensor)
    if not np.isfinite(p_s).all() or p_s[0] <= MIN_DEPTH:
        raise ValueError(
            "Camera projection needs finite coordinates and depth "
            f"> {MIN_DEPTH:g}; sensor position={p_s.tolist()}"
        )
    x_s, y_s, z_s = p_s
    u = sensor.c_i - sensor.f_i * y_s / x_s
    v = sensor.c_j - sensor.f_j * z_s / x_s
    return np.asmatrix([[u], [v]], dtype=float)


def build_camera_measurement(z: Sequence[float], sensor: Any) -> dict[str, Any]:
    """Build camera measurement vector z and covariance R from pixel coordinates.

    Args:
        z: Sequence ``[u, v]`` pixel coordinates.
        sensor: Camera sensor object.

    Returns:
        Dict with keys ``z``, ``R``, ``sensor``.
    """
    # vi: TODO Part G — z mat 2x1; R diag sigma_cam_i^2, sigma_cam_j^2 từ params.
    params = get_tracking_params()
    z_mat = np.asmatrix(np.asarray(z, dtype=float).reshape(-1)[:2]).T
    R = np.asmatrix(np.diag([params.sigma_cam_i**2, params.sigma_cam_j**2]))
    return {"z": z_mat, "R": R, "sensor": sensor}
