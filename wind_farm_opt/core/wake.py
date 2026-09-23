"""尾流模型实现。

包含 Jensen 模型和高斯剖面尾流模型，以及多尾流叠加（平方和）。

模型契约
--------
所有尾流模型统一遵守：

- ``velocity_deficit`` 返回中心线（峰值）速度亏损 ``1 - u/U0``，取值 [0, 1)；
- ``wake_radius`` 返回给定下游距离处的尾流特征半径（恒为正）；
- ``radial_profile`` 返回 [0, 1] 的径向衰减系数，中心线处为 1；
- 单台风机在任意空间点上产生的有效亏损统一通过
  :func:`wake_deficit_from_turbine` 计算，交互明细、AEP 叠加和热力图
  不得各自重新实现几何投影，避免对同一契约产生不一致的解释。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# 推力系数允许的物理范围。超出该范围的值会被截断到边界；
# 仅精确等于 1.0 时高斯模型内部需要用略小于 1 的值避免除零。
_CT_MIN = 0.0
_CT_MAX = 1.0
_CT_SAFE_MAX = 1.0 - 1e-12


def _as_float_array(value) -> np.ndarray:
    """转为 float64 数组（标量成为 0 维数组）。"""
    return np.asarray(value, dtype=np.float64)


def _unwrap(result: np.ndarray, reference) -> float | np.ndarray:
    """参考输入为标量时把结果还原成 Python float。"""
    ref = np.asarray(reference)
    return float(result) if ref.ndim == 0 else result


def _sanitize_thrust_coefficient(thrust_coefficient) -> np.ndarray:
    """清洗推力系数。

    非物理值（Ct<0 或 Ct>1）截断到 [0, 1]；NaN / inf 没有物理含义，
    直接抛出异常而不是悄悄污染整场亏损。
    """
    ct = _as_float_array(thrust_coefficient)
    if not np.all(np.isfinite(ct)):
        raise ValueError(f"推力系数必须为有限数值，当前为 {thrust_coefficient}")
    return np.clip(ct, _CT_MIN, _CT_MAX)


def _validate_rotor_diameter(rotor_diameter) -> np.ndarray:
    d0 = _as_float_array(rotor_diameter)
    if np.any(d0 <= 0.0):
        raise ValueError(f"转子直径必须为正数，当前为 {rotor_diameter}")
    return d0


class WakeModel(ABC):
    """尾流模型抽象基类。"""

    @abstractmethod
    def velocity_deficit(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float,
        thrust_coefficient: float,
    ) -> float | np.ndarray:
        """计算尾流中心线（峰值）速度亏损。

        Parameters
        ----------
        distance : float | np.ndarray
            下游距离 (m)，非正距离视为不在尾流中（亏损为 0）
        rotor_diameter : float
            上游风机转子直径 (m)，必须为正
        thrust_coefficient : float
            上游风机推力系数，超出 [0, 1] 的值会被截断

        Returns
        -------
        float | np.ndarray
            中心线速度亏损 (1 - u/U0)，范围 [0, 1)，
            对 Jensen 模型随下游距离单调递减并在远场趋于 0
        """
        pass

    @abstractmethod
    def wake_radius(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float,
    ) -> float | np.ndarray:
        """计算尾流特征半径。

        Parameters
        ----------
        distance : float | np.ndarray
            下游距离 (m)
        rotor_diameter : float
            上游风机转子直径 (m)，必须为正

        Returns
        -------
        float | np.ndarray
            尾流半径 (m)，恒为正
        """
        pass

    @abstractmethod
    def radial_profile(
        self,
        radial_dist: float | np.ndarray,
        wake_radius: float,
    ) -> float | np.ndarray:
        """计算径向速度亏损分布。

        Parameters
        ----------
        radial_dist : float | np.ndarray
            到尾流中心线的径向距离 (m)
        wake_radius : float
            尾流半径 (m)

        Returns
        -------
        float | np.ndarray
            径向分布系数，范围 [0, 1]，中心线处为 1
        """
        pass


class JensenWake(WakeModel):
    """Jensen 尾流模型 (1983)。

    经典的锥形尾流模型，假设尾流线性扩张，速度亏损在尾流截面均匀分布。

    中心线亏损::

        delta(x) = (1 - sqrt(1 - Ct)) * (D / (D + 2 k x)) ** 2

    其中 ``x`` 为下游距离。``x = 0`` 处为转子平面感应亏损
    ``1 - sqrt(1 - Ct)``，随 ``x`` 增大单调递减并趋于 0。
    """

    def __init__(self, wake_decay: float = 0.07) -> None:
        if wake_decay < 0.0:
            raise ValueError(f"尾流衰减系数不能为负，当前为 {wake_decay}")
        self.wake_decay = wake_decay

    def velocity_deficit(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float,
        thrust_coefficient: float,
    ) -> float | np.ndarray:
        dist = _as_float_array(distance)
        d0 = _validate_rotor_diameter(rotor_diameter)
        ct = _sanitize_thrust_coefficient(thrust_coefficient)

        downstream = dist > 0.0
        # 负距离取 0 仅用于让展开比保持有界；最终结果会按下游掩码清零。
        x = np.maximum(dist, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            expansion_ratio = d0 / (d0 + 2.0 * self.wake_decay * x)
            rotor_plane_deficit = 1.0 - np.sqrt(np.maximum(1.0 - ct, 0.0))
            deficit = rotor_plane_deficit * expansion_ratio ** 2

        deficit = np.where(downstream, deficit, 0.0)
        deficit = np.clip(deficit, 0.0, 1.0)

        return _unwrap(deficit, distance)

    def wake_radius(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float,
    ) -> float | np.ndarray:
        dist = _as_float_array(distance)
        d0 = _validate_rotor_diameter(rotor_diameter)
        # 尾流从转子半径开始线性扩张；非正距离（含零距离）取转子半径。
        radius = 0.5 * d0 + self.wake_decay * np.maximum(dist, 0.0)
        return _unwrap(radius, distance)

    def radial_profile(
        self,
        radial_dist: float | np.ndarray,
        wake_radius: float,
    ) -> float | np.ndarray:
        r = _as_float_array(radial_dist)
        radius = _as_float_array(wake_radius)
        # 顶帽剖面：尾流内部亏损均匀，外部为 0。
        profile = np.where(np.abs(r) <= radius, 1.0, 0.0)
        return _unwrap(profile, radial_dist)


class GaussianWake(WakeModel):
    """Bastankhah & Porté-Agel 高斯剖面尾流模型 (2014)。

    假设尾流速度亏损符合高斯分布，更符合实际风洞和实测数据。
    仅在近尾流长度 ``x0`` 之外严格成立；更近的位置按 ``x0`` 处取值，
    并用 0 亏损表示模型未覆盖的上游/重合位置。
    """

    def __init__(
        self,
        wake_decay: float = 0.035,
        near_wake_length: float = 3.0,
    ) -> None:
        if wake_decay < 0.0:
            raise ValueError(f"尾流衰减系数不能为负，当前为 {wake_decay}")
        if near_wake_length < 0.0:
            raise ValueError(f"近尾流长度不能为负，当前为 {near_wake_length}")
        self.wake_decay = wake_decay
        self.near_wake_length = near_wake_length

    def velocity_deficit(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float,
        thrust_coefficient: float,
    ) -> float | np.ndarray:
        dist = _as_float_array(distance)
        d0 = _validate_rotor_diameter(rotor_diameter)
        ct = _sanitize_thrust_coefficient(thrust_coefficient)
        # Ct 精确为 1 时 sqrt(1-Ct)=0 会让 beta 除零；截断到略小于 1，
        # 对所有 Ct < 1 的结果没有影响。
        ct_safe = np.minimum(ct, _CT_SAFE_MAX)

        beta = 0.5 * (1.0 + np.sqrt(1.0 - ct_safe)) / np.sqrt(1.0 - ct_safe)

        x_nd = np.maximum(dist, 0.0) / d0
        x_over = np.maximum(x_nd, self.near_wake_length)
        sigma = self.wake_decay * x_over * d0 + d0 / 2.0 / np.sqrt(beta)

        peak_deficit = 1.0 - np.sqrt(
            np.maximum(1.0 - ct / (8.0 * (sigma / d0) ** 2), 0.0)
        )

        peak_deficit = np.where(dist > 0.0, peak_deficit, 0.0)
        peak_deficit = np.clip(peak_deficit, 0.0, 1.0)

        return _unwrap(peak_deficit, distance)

    def wake_radius(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float,
    ) -> float | np.ndarray:
        dist = _as_float_array(distance)
        d0 = _validate_rotor_diameter(rotor_diameter)

        x_nd = np.maximum(dist, 0.0) / d0
        x_over = np.maximum(x_nd, self.near_wake_length)
        sigma = self.wake_decay * x_over * d0 + d0 / 4.0

        radius = 2.0 * sigma
        radius = np.where(dist > 0.0, radius, 0.5 * d0)

        return _unwrap(radius, distance)

    def radial_profile(
        self,
        radial_dist: float | np.ndarray,
        wake_radius: float,
    ) -> float | np.ndarray:
        r = _as_float_array(radial_dist)
        sigma = _as_float_array(wake_radius) / 2.0

        with np.errstate(divide="ignore", invalid="ignore"):
            profile = np.exp(-0.5 * (r / sigma) ** 2)

        # 退化尾流（半径非正）下不产生亏损，包括中心线点。
        profile = np.where(sigma > 0.0, profile, 0.0)

        return _unwrap(profile, radial_dist)


def superpose_wakes(
    deficits: np.ndarray,
    method: str = "sum_of_squares",
) -> np.ndarray:
    """叠加多个尾流的速度亏损。

    Parameters
    ----------
    deficits : np.ndarray
        每个上游风机产生的速度亏损数组，形状为 (N_upstream, ...)；
        允许为空数组，此时总亏损为 0
    method : str
        叠加方法："sum_of_squares"（平方和，推荐）或 "linear"（线性叠加）

    Returns
    -------
    np.ndarray
        叠加后的总速度亏损，形状为 (...)，取值 [0, 1]
    """
    deficits = np.asarray(deficits, dtype=np.float64)

    if deficits.size == 0:
        return np.zeros(np.asarray(deficits).shape[1:], dtype=np.float64)

    if not np.all(np.isfinite(deficits)):
        raise ValueError("参与叠加的尾流亏损必须全部为有限数值")

    if method == "sum_of_squares":
        total_deficit = np.sqrt(np.sum(deficits ** 2, axis=0))
    elif method == "linear":
        total_deficit = np.sum(deficits, axis=0)
    else:
        raise ValueError(f"未知的尾流叠加方法: {method}")

    return np.clip(total_deficit, 0.0, 1.0)


def wake_deficit_from_turbine(
    wake_model: WakeModel,
    upstream_position: np.ndarray,
    eval_points: np.ndarray,
    wind_vector: np.ndarray,
    rotor_diameter: float,
    thrust_coefficient: float,
) -> np.ndarray:
    """计算单台上游风机在一个或多个空间点上产生的有效速度亏损。

    这是尾流模型契约的唯一几何入口：风向投影、下游/横向距离分解、
    中心线亏损与径向剖面的组合都在这里完成，交互明细、全场叠加和
    热力图共用本函数，保证三处结果方向和量级一致。

    Parameters
    ----------
    wake_model : WakeModel
        尾流模型实例
    upstream_position : np.ndarray
        上游风机坐标，形状 (2,)
    eval_points : np.ndarray
        待评估点坐标，形状 (..., 2)，可以是单个点或二维网格
    wind_vector : np.ndarray
        单位风向向量，形状 (2,)
    rotor_diameter : float
        上游风机转子直径 (m)
    thrust_coefficient : float
        上游风机推力系数

    Returns
    -------
    np.ndarray
        各评估点处的有效速度亏损 (1 - u/U0)，形状与 eval_points 的
        前导维度一致；重合点与上游侧点亏损为 0
    """
    points = _as_float_array(eval_points)
    source = _as_float_array(upstream_position)
    wind = _as_float_array(wind_vector)

    single_point = points.ndim == 1
    if single_point:
        points = points[np.newaxis, :]

    delta = points - source
    distance = np.linalg.norm(delta, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_norm = np.where(
            distance[..., np.newaxis] > 1e-12,
            delta / distance[..., np.newaxis],
            0.0,
        )

    along_wind = np.sum(delta_norm * wind, axis=-1)
    downstream_mask = (along_wind > 0.0) & (distance > 1e-12)

    downstream_dist = np.where(downstream_mask, distance * along_wind, 0.0)
    cross_dist = np.where(
        downstream_mask,
        distance * np.sqrt(np.clip(1.0 - along_wind ** 2, 0.0, 1.0)),
        0.0,
    )

    radius = wake_model.wake_radius(downstream_dist, rotor_diameter)
    peak = wake_model.velocity_deficit(
        downstream_dist, rotor_diameter, thrust_coefficient
    )
    radial = wake_model.radial_profile(cross_dist, radius)

    deficit = np.where(downstream_mask, peak * radial, 0.0)
    deficit = np.clip(deficit, 0.0, 1.0)

    if single_point:
        deficit = deficit[0]
    return deficit


def wake_deficit_field(
    wake_model: WakeModel,
    turbine_positions: np.ndarray,
    eval_points: np.ndarray,
    wind_direction: float,
    rotor_diameters: np.ndarray,
    thrust_coefficients: np.ndarray,
    method: str = "sum_of_squares",
) -> np.ndarray:
    """计算全部风机在一组空间点上叠加后的总速度亏损场。

    单台风机贡献由 :func:`wake_deficit_from_turbine` 给出，多台风机按
    ``method`` 叠加。全场 AEP 与热力图共用本函数。

    Parameters
    ----------
    wake_model : WakeModel
        尾流模型实例
    turbine_positions : np.ndarray
        风机位置，形状 (N_turb, 2)
    eval_points : np.ndarray
        评估点，形状 (..., 2)
    wind_direction : float
        气象风向 (度)
    rotor_diameters, thrust_coefficients : np.ndarray
        每台风机的转子直径与推力系数，形状 (N_turb,)
    method : str
        叠加方法，见 :func:`superpose_wakes`

    Returns
    -------
    np.ndarray
        叠加亏损场，形状为 eval_points 的前导维度，取值 [0, 1]
    """
    positions = _as_float_array(turbine_positions)
    diameters = _as_float_array(rotor_diameters)
    cts = _sanitize_thrust_coefficient(thrust_coefficients)
    wind_vec = wind_unit_vector(wind_direction)

    per_turbine = np.empty((positions.shape[0],) + eval_points.shape[:-1])
    for i in range(positions.shape[0]):
        per_turbine[i] = wake_deficit_from_turbine(
            wake_model,
            positions[i],
            eval_points,
            wind_vec,
            diameters[i],
            cts[i],
        )

    return superpose_wakes(per_turbine, method=method)


@dataclass
class WakeInteraction:
    """尾流相互作用结果。

    Parameters
    ----------
    upstream_idx : int
        上游风机索引
    downstream_idx : int
        下游风机索引
    distance : float
        两台风机之间的距离 (m)
    angle_from_wind : float
        两台风机连线与风向的夹角 (度)
    velocity_deficit : float
        速度亏损值
    affected_power : float
        受影响的功率损失 (kW)
    in_wake : bool
        是否在尾流影响范围内
    """

    upstream_idx: int
    downstream_idx: int
    distance: float
    angle_from_wind: float
    velocity_deficit: float
    affected_power: float
    in_wake: bool


def wind_unit_vector(wind_direction: float) -> np.ndarray:
    """气象风向（0度为北、顺时针）对应的单位流向向量。"""
    wind_rad = np.deg2rad(270.0 - wind_direction)
    return np.array([np.cos(wind_rad), np.sin(wind_rad)])


def compute_wake_interactions(
    positions: np.ndarray,
    wind_direction: float,
    rotor_diameters: np.ndarray,
    thrust_coefficients: np.ndarray,
    wake_model: WakeModel,
    power_curves: list[np.ndarray],
    free_stream_speed: float,
) -> list[WakeInteraction]:
    """计算给定风向下的所有尾流相互作用。

    Parameters
    ----------
    positions : np.ndarray
        风机位置，形状为 (N_turbines, 2)
    wind_direction : float
        风向 (度)，0度为北，顺时针
    rotor_diameters : np.ndarray
        每台风机的转子直径，形状为 (N_turbines,)
    thrust_coefficients : np.ndarray
        每台风机的推力系数，形状为 (N_turbines,)
    wake_model : WakeModel
        尾流模型实例
    power_curves : list[np.ndarray]
        每台风机的功率曲线
    free_stream_speed : float
        自由来流风速 (m/s)

    Returns
    -------
    list[WakeInteraction]
        尾流相互作用列表
    """
    n = positions.shape[0]
    interactions = []

    wind_vec = wind_unit_vector(wind_direction)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            delta = positions[j] - positions[i]
            distance = float(np.linalg.norm(delta))

            if distance <= 1e-12:
                continue

            along_wind = float(np.dot(delta / distance, wind_vec))
            if along_wind <= 0.0:
                continue

            deficit = float(
                wake_deficit_from_turbine(
                    wake_model,
                    positions[i],
                    positions[j],
                    wind_vec,
                    rotor_diameters[i],
                    thrust_coefficients[i],
                )
            )

            angle_deg = float(np.rad2deg(np.arccos(np.clip(along_wind, -1.0, 1.0))))

            effective_speed = free_stream_speed * (1.0 - deficit)
            power_free = np.interp(free_stream_speed, power_curves[j][:, 0], power_curves[j][:, 1],
                                  left=0.0, right=0.0)
            power_wake = np.interp(effective_speed, power_curves[j][:, 0], power_curves[j][:, 1],
                                  left=0.0, right=0.0)
            power_loss = power_free - power_wake

            in_wake = deficit > 0.001

            interactions.append(
                WakeInteraction(
                    upstream_idx=i,
                    downstream_idx=j,
                    distance=distance,
                    angle_from_wind=angle_deg,
                    velocity_deficit=deficit,
                    affected_power=float(power_loss),
                    in_wake=bool(in_wake),
                )
            )

    return interactions
