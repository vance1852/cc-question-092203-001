"""尾流模型实现。

包含 Jensen 模型和高斯剖面尾流模型，以及多尾流叠加（平方和）。

模型契约
--------
所有 :class:`WakeModel` 子类遵守同一契约，单对交互计算
(:func:`compute_wake_interactions`)、全场 AEP 叠加
(:mod:`wind_farm_opt.farm.aep`) 和热力图
(:func:`wind_farm_opt.visualization.plotting.plot_wake_heatmap`)
均通过本模块提供的共享函数使用该契约，不得各自重新实现：

- ``velocity_deficit`` 返回尾流**中心线**速度亏损 ``ΔU/U0 = 1 - u/U0``，
  取值 ``[0, 1)``；距离 ``x <= 0``（风机自身位置或上游）时亏损为 0；
  Jensen 的中心线亏损随下游距离单调衰减。
- ``wake_radius`` 返回尾流半径，``x <= 0`` 时等于转子半径 ``D/2``。
- ``radial_profile`` 返回 ``[0, 1]`` 的径向系数，中心线（r=0）为 1。
- 空间任意点的有效亏损 = ``velocity_deficit * radial_profile``。
- 多个上游尾流按 :func:`superpose_wakes` 叠加（默认平方和）。

标量（含 Python float/int）输入返回 Python ``float``，数组输入返回
同广播形状的 ``np.ndarray``。非物理推力系数会被裁剪到 ``[0, 1]``，
模型在任何输入下都不返回 ``NaN``/``inf``。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

#: 判定两点重合 / 风机处于下游的几何容差
_EPS_DISTANCE = 1e-12


class WakeModel(ABC):
    """尾流模型抽象基类。"""

    @abstractmethod
    def velocity_deficit(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float | np.ndarray,
        thrust_coefficient: float | np.ndarray,
    ) -> float | np.ndarray:
        """计算尾流中心线轴向速度亏损。

        Parameters
        ----------
        distance : float | np.ndarray
            下游距离 (m)，必须 >= 0；``x <= 0`` 处亏损定义为 0
        rotor_diameter : float | np.ndarray
            上游风机转子直径 (m)，必须 > 0
        thrust_coefficient : float | np.ndarray
            上游风机推力系数 Ct，超出 ``[0, 1]`` 的非物理值裁剪到边界

        Returns
        -------
        float | np.ndarray
            中心线速度亏损 ``1 - u/U0``，范围 ``[0, 1)``
        """
        pass

    @abstractmethod
    def wake_radius(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float | np.ndarray,
    ) -> float | np.ndarray:
        """计算尾流半径。

        Parameters
        ----------
        distance : float | np.ndarray
            下游距离 (m)
        rotor_diameter : float | np.ndarray
            上游风机转子直径 (m)

        Returns
        -------
        float | np.ndarray
            尾流半径 (m)，``x <= 0`` 时为转子半径 ``D/2``
        """
        pass

    @abstractmethod
    def radial_profile(
        self,
        radial_dist: float | np.ndarray,
        wake_radius: float | np.ndarray,
    ) -> float | np.ndarray:
        """计算径向速度亏损分布系数。

        Parameters
        ----------
        radial_dist : float | np.ndarray
            到尾流中心线的径向距离 (m)
        wake_radius : float | np.ndarray
            尾流半径 (m)，支持数组与 ``radial_dist`` 广播

        Returns
        -------
        float | np.ndarray
            径向分布系数，范围 ``[0, 1]``，中心线处为 1
        """
        pass


def _sanitize_inputs(
    distance: float | np.ndarray,
    rotor_diameter: float | np.ndarray,
    thrust_coefficient: float | np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """转为 float64 数组、广播形状，并把 Ct 裁剪到物理区间 [0, 1]。"""
    dist = np.asarray(distance, dtype=np.float64)
    diameter = np.asarray(rotor_diameter, dtype=np.float64)
    if thrust_coefficient is None:
        dist, diameter = np.broadcast_arrays(dist, diameter)
        return dist, diameter

    ct = np.clip(np.asarray(thrust_coefficient, dtype=np.float64), 0.0, 1.0)
    dist, diameter, ct = np.broadcast_arrays(dist, diameter, ct)
    return dist, diameter, ct


def _scalar_or_array(result: np.ndarray) -> float | np.ndarray:
    """0 维结果返回 Python float，其余返回数组。"""
    return float(result) if result.ndim == 0 else result


class JensenWake(WakeModel):
    """Jensen 尾流模型 (1983, PARK 模型)。

    经典的锥形尾流模型，假设尾流线性扩张，速度亏损在尾流截面均匀分布。

    中心线速度亏损::

        ΔU/U0 = (1 - sqrt(1 - Ct)) / (1 + 2*k*x/D)^2

    随下游距离 ``x`` 单调衰减，远场趋于 0。
    """

    def __init__(self, wake_decay: float = 0.07) -> None:
        if wake_decay < 0.0:
            raise ValueError(f"尾流衰减系数必须非负，当前为 {wake_decay}")
        self.wake_decay = wake_decay

    def velocity_deficit(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float | np.ndarray,
        thrust_coefficient: float | np.ndarray,
    ) -> float | np.ndarray:
        dist, d0, ct = _sanitize_inputs(distance, rotor_diameter, thrust_coefficient)

        # 转子平面处的最大亏损（轴向诱导因子形式）
        rotor_plane_deficit = 1.0 - np.sqrt(1.0 - ct)

        with np.errstate(divide="ignore", invalid="ignore"):
            deficit = rotor_plane_deficit / (
                1.0 + 2.0 * self.wake_decay * dist / d0
            ) ** 2

        # 风机自身位置/上游方向及非物理直径处不产生亏损
        valid = (dist > 0.0) & np.isfinite(dist) & (d0 > 0.0)
        deficit = np.where(valid, deficit, 0.0)
        deficit = np.nan_to_num(deficit, nan=0.0, posinf=1.0, neginf=0.0)
        deficit = np.clip(deficit, 0.0, 1.0)

        return _scalar_or_array(deficit)

    def wake_radius(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float | np.ndarray,
    ) -> float | np.ndarray:
        dist, d0 = _sanitize_inputs(distance, rotor_diameter)
        downstream = np.where(dist > 0.0, dist, 0.0)
        radius = 0.5 * d0 + self.wake_decay * downstream
        radius = np.where(d0 > 0.0, radius, 0.0)
        return _scalar_or_array(radius)

    def radial_profile(
        self,
        radial_dist: float | np.ndarray,
        wake_radius: float | np.ndarray,
    ) -> float | np.ndarray:
        r = np.asarray(radial_dist, dtype=np.float64)
        wr = np.asarray(wake_radius, dtype=np.float64)
        profile = np.where(np.abs(r) <= wr, 1.0, 0.0)
        return _scalar_or_array(profile)


class GaussianWake(WakeModel):
    """Bastankhah & Porté-Agel 高斯剖面尾流模型 (2014)。

    假设尾流速度亏损符合高斯分布，更符合实际风洞和实测数据。
    中心线亏损::

        ΔU/U0 = 1 - sqrt(1 - Ct / (8*(σ/D)^2))

    近场（``x < near_wake_length * D``）内 σ 冻结在近场长度处，
    因此中心线亏损在近场区保持平台值。
    """

    def __init__(
        self,
        wake_decay: float = 0.035,
        near_wake_length: float = 3.0,
    ) -> None:
        if wake_decay < 0.0:
            raise ValueError(f"尾流衰减系数必须非负，当前为 {wake_decay}")
        if near_wake_length <= 0.0:
            raise ValueError(
                f"近场长度必须为正，当前为 {near_wake_length}"
            )
        self.wake_decay = wake_decay
        self.near_wake_length = near_wake_length

    def velocity_deficit(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float | np.ndarray,
        thrust_coefficient: float | np.ndarray,
    ) -> float | np.ndarray:
        dist, d0, ct = _sanitize_inputs(distance, rotor_diameter, thrust_coefficient)

        with np.errstate(divide="ignore", invalid="ignore"):
            beta = 0.5 * (1.0 + np.sqrt(1.0 - ct)) / np.sqrt(1.0 - ct)

            x_nd = dist / d0
            x_over = np.maximum(x_nd, self.near_wake_length)
            sigma = self.wake_decay * x_over * d0 + d0 / 2.0 / np.sqrt(beta)

            # Ct=1 且 σ→0 时分母失效；将根号内参数裁剪到 [0, 1]，
            # 此时亏损取物理上限 1，而不是 NaN。
            inner = ct / (8.0 * (sigma / d0) ** 2)
            inner = np.where(sigma > 0.0, np.clip(inner, 0.0, 1.0), 1.0)
            peak_deficit = 1.0 - np.sqrt(1.0 - inner)

        valid = (dist > 0.0) & np.isfinite(dist) & (d0 > 0.0)
        peak_deficit = np.where(valid, peak_deficit, 0.0)
        peak_deficit = np.nan_to_num(peak_deficit, nan=0.0, posinf=1.0)
        peak_deficit = np.clip(peak_deficit, 0.0, 1.0)

        return _scalar_or_array(peak_deficit)

    def wake_radius(
        self,
        distance: float | np.ndarray,
        rotor_diameter: float | np.ndarray,
    ) -> float | np.ndarray:
        dist, d0 = _sanitize_inputs(distance, rotor_diameter)

        x_nd = dist / d0
        x_over = np.maximum(x_nd, self.near_wake_length)
        sigma = self.wake_decay * x_over * d0 + d0 / 4.0

        radius = 2.0 * sigma
        radius = np.where(dist > 0.0, radius, 0.5 * d0)
        radius = np.where(d0 > 0.0, radius, 0.0)

        return _scalar_or_array(radius)

    def radial_profile(
        self,
        radial_dist: float | np.ndarray,
        wake_radius: float | np.ndarray,
    ) -> float | np.ndarray:
        r = np.asarray(radial_dist, dtype=np.float64)
        wr = np.asarray(wake_radius, dtype=np.float64)
        sigma = wr / 2.0

        with np.errstate(divide="ignore", invalid="ignore"):
            profile = np.exp(-0.5 * (r / sigma) ** 2)

        profile = np.where(sigma > 0.0, profile, 0.0)
        return _scalar_or_array(profile)


def superpose_wakes(
    deficits: np.ndarray,
    method: str = "sum_of_squares",
) -> np.ndarray:
    """叠加多个尾流的速度亏损。

    Parameters
    ----------
    deficits : np.ndarray
        每个上游风机产生的速度亏损数组，形状为 (N_upstream, ...)；
        N_upstream 为 0（无上游风机）时总亏损为 0
    method : str
        叠加方法："sum_of_squares"（平方和，推荐）或 "linear"（线性叠加）

    Returns
    -------
    np.ndarray
        叠加后的总速度亏损，形状为 (...)，裁剪到 ``[0, 1]``
    """
    deficits = np.asarray(deficits, dtype=np.float64)

    if method == "sum_of_squares":
        total_deficit = np.sqrt(np.sum(deficits ** 2, axis=0))
    elif method == "linear":
        total_deficit = np.sum(deficits, axis=0)
    else:
        raise ValueError(f"未知的尾流叠加方法: {method}")

    total_deficit = np.nan_to_num(total_deficit, nan=0.0, posinf=1.0)
    return np.clip(total_deficit, 0.0, 1.0)


def wind_unit_vector(wind_direction: float) -> np.ndarray:
    """风向（度，0=北顺时针）对应的风传播方向单位向量。"""
    wind_rad = np.deg2rad(270.0 - wind_direction)
    return np.array([np.cos(wind_rad), np.sin(wind_rad)])


def wake_geometry(
    delta: np.ndarray,
    wind_direction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算相对位移向量相对风向的尾流几何量。

    单对交互、AEP 矩阵和热力图共用本函数，保证三条路径的方向
    约定与数值完全一致。

    Parameters
    ----------
    delta : np.ndarray
        从上风机指向受影点的位移，形状 (..., 2)
    wind_direction : float
        风向 (度)，0 度为北，顺时针

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - downstream_dist : 下游轴向距离 (m)，非下游点为 0
        - cross_dist : 横向偏移 (m)，非下游点为 0
        - downstream_mask : 受影点是否严格位于风机下游
    """
    delta = np.asarray(delta, dtype=np.float64)
    wind_vec = wind_unit_vector(wind_direction)

    distance = np.linalg.norm(delta, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_norm = np.where(
            distance[..., np.newaxis] > _EPS_DISTANCE,
            delta / distance[..., np.newaxis],
            0.0,
        )

    along_wind = np.sum(delta_norm * wind_vec, axis=-1)

    downstream_mask = (along_wind > 0.0) & (distance > _EPS_DISTANCE)

    downstream_dist = np.where(downstream_mask, distance * along_wind, 0.0)
    cross_dist = np.where(
        downstream_mask,
        distance * np.sqrt(np.clip(1.0 - along_wind ** 2, 0.0, 1.0)),
        0.0,
    )
    return downstream_dist, cross_dist, downstream_mask


def point_velocity_deficit(
    downstream_dist: float | np.ndarray,
    cross_dist: float | np.ndarray,
    rotor_diameter: float | np.ndarray,
    thrust_coefficient: float | np.ndarray,
    wake_model: WakeModel,
) -> float | np.ndarray:
    """单台上游风机在空间点上的有效速度亏损（中心线亏损 × 径向系数）。

    这是模型契约中“空间点亏损”的唯一实现，所有调用方都应经由本函数，
    以保证交互明细、AEP 与热力图的量级一致。
    """
    wr = wake_model.wake_radius(downstream_dist, rotor_diameter)
    peak_deficit = wake_model.velocity_deficit(
        downstream_dist, rotor_diameter, thrust_coefficient
    )
    radial_factor = wake_model.radial_profile(cross_dist, wr)
    return peak_deficit * radial_factor


def compute_pairwise_deficits(
    positions: np.ndarray,
    wind_direction: float,
    rotor_diameters: np.ndarray,
    thrust_coefficients: np.ndarray,
    wake_model: WakeModel,
) -> np.ndarray:
    """计算所有风机对之间（未叠加）的速度亏损矩阵。

    Parameters
    ----------
    positions : np.ndarray
        风机位置 (N, 2)
    wind_direction : float
        风向 (度)
    rotor_diameters : np.ndarray
        转子直径 (N,)
    thrust_coefficients : np.ndarray
        推力系数 (N,)
    wake_model : WakeModel
        尾流模型实例

    Returns
    -------
    np.ndarray
        亏损矩阵 (N_upstream, N_downstream)，对角线与非下游对为 0。
        对列调用 :func:`superpose_wakes` 即得到每台风机承受的总亏损。
    """
    positions = np.asarray(positions, dtype=np.float64)
    diameters = np.asarray(rotor_diameters, dtype=np.float64)
    cts = np.asarray(thrust_coefficients, dtype=np.float64)

    # delta[i, j] = 从风机 i 指向风机 j
    delta = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    downstream_dist, cross_dist, mask = wake_geometry(delta, wind_direction)

    deficit = point_velocity_deficit(
        downstream_dist,
        cross_dist,
        diameters[:, np.newaxis],
        cts[:, np.newaxis],
        wake_model,
    )
    return np.where(mask, deficit, 0.0)


def compute_deficit_field(
    points: np.ndarray,
    positions: np.ndarray,
    wind_direction: float,
    rotor_diameters: np.ndarray,
    thrust_coefficients: np.ndarray,
    wake_model: WakeModel,
    method: str = "sum_of_squares",
) -> np.ndarray:
    """计算一组空间点上由全部风机叠加产生的总速度亏损。

    热力图等网格计算与 AEP 使用同一个 :func:`superpose_wakes` 契约，
    不再自行实现平方和。

    Parameters
    ----------
    points : np.ndarray
        受影点位置 (M, 2)
    positions : np.ndarray
        风机位置 (N, 2)
    wind_direction : float
        风向 (度)
    rotor_diameters, thrust_coefficients : np.ndarray
        每台风机的转子直径 (N,) 与推力系数 (N,)
    wake_model : WakeModel
        尾流模型实例
    method : str
        叠加方法，见 :func:`superpose_wakes`

    Returns
    -------
    np.ndarray
        每个受影点的总速度亏损 (M,)
    """
    points = np.asarray(points, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    diameters = np.asarray(rotor_diameters, dtype=np.float64)
    cts = np.asarray(thrust_coefficients, dtype=np.float64)

    # delta[i, m] = 从风机 i 指向受影点 m
    delta = points[np.newaxis, :, :] - positions[:, np.newaxis, :]
    downstream_dist, cross_dist, mask = wake_geometry(delta, wind_direction)

    deficits = point_velocity_deficit(
        downstream_dist,
        cross_dist,
        diameters[:, np.newaxis],
        cts[:, np.newaxis],
        wake_model,
    )
    deficits = np.where(mask, deficits, 0.0)
    return superpose_wakes(deficits, method=method)


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

    与全场 AEP、热力图共用 :func:`wake_geometry` 和
    :func:`point_velocity_deficit`，因此同一布局的单对亏损明细与
    叠加场、热力图在方向和量级上保持一致。

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

            if distance <= _EPS_DISTANCE:
                continue

            downstream_dist, cross_dist, is_downstream = wake_geometry(
                delta, wind_direction
            )
            if not bool(is_downstream):
                continue

            wr = wake_model.wake_radius(float(downstream_dist), rotor_diameters[i])
            radial_factor = float(
                wake_model.radial_profile(float(cross_dist), wr)
            )
            peak_deficit = float(
                wake_model.velocity_deficit(
                    float(downstream_dist),
                    rotor_diameters[i],
                    thrust_coefficients[i],
                )
            )
            deficit = peak_deficit * radial_factor

            along_wind = float(np.dot(delta / distance, wind_vec))
            angle_deg = float(np.rad2deg(np.arccos(np.clip(along_wind, -1.0, 1.0))))

            effective_speed = free_stream_speed * (1.0 - deficit)
            power_free = np.interp(free_stream_speed, power_curves[j][:, 0], power_curves[j][:, 1],
                                  left=0.0, right=0.0)
            power_wake = np.interp(effective_speed, power_curves[j][:, 0], power_curves[j][:, 1],
                                  left=0.0, right=0.0)
            power_loss = power_free - power_wake

            in_wake = radial_factor > 0.01 and deficit > 0.001

            interactions.append(
                WakeInteraction(
                    upstream_idx=i,
                    downstream_idx=j,
                    distance=distance,
                    angle_from_wind=angle_deg,
                    velocity_deficit=float(deficit),
                    affected_power=float(power_loss),
                    in_wake=bool(in_wake),
                )
            )

    return interactions
