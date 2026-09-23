"""AEP 计算与尾流契约的集成回归测试。"""

import numpy as np
import pytest

from wind_farm_opt.core.turbine import create_default_turbine
from wind_farm_opt.core.wind_resource import (
    WindResource,
    WindSector,
    create_simple_wind_resource,
    _gamma_lanczos,
)
from wind_farm_opt.core.wake import (
    JensenWake,
    GaussianWake,
    compute_pairwise_deficits,
)
from wind_farm_opt.farm.aep import AEPCalculator


def _single_direction_resource(direction=270.0, mean_speed=8.5, k=2.1):
    """风向严格对齐给定方向的单扇区资源，用于同轴尾流测试。"""
    c = mean_speed / float(_gamma_lanczos(np.array(1.0 + 1.0 / k)))
    return WindResource([
        WindSector(
            direction_center=direction,
            direction_width=360.0,
            frequency=1.0,
            mean_speed=mean_speed,
            weibull_k=k,
            weibull_c=float(c),
        )
    ])


def _make_calculator(model, n_turb=4):
    turbines = [create_default_turbine("V126-3.45MW") for _ in range(n_turb)]
    wind_resource = create_simple_wind_resource(
        num_sectors=4, uniform=True, mean_speed=8.5
    )
    return AEPCalculator(
        turbines=turbines,
        wind_resource=wind_resource,
        wake_model=model,
        wake_superposition="sum_of_squares",
        speed_step=1.0,
    )


@pytest.mark.parametrize("model", [JensenWake(0.07), GaussianWake(0.035)])
def test_aep_deficit_matrix_matches_shared_contract(model):
    calc = _make_calculator(model)
    positions = np.array([
        [0.0, 0.0],
        [7.0 * 126.0, 0.0],
        [0.0, 7.0 * 126.0],
        [7.0 * 126.0, 7.0 * 126.0],
    ])

    # 主西风（扇区中心 270 附近）下 AEP 内部使用的两两矩阵
    # 必须与 core.wake 的统一契约给出相同结果
    direction = calc.wind_resource.directions[
        int(np.argmin(np.abs(calc.wind_resource.directions - 270.0)))
    ]
    internal_total, internal_matrix = calc._compute_wake_deficit_field(
        positions, direction
    )
    ref_matrix = compute_pairwise_deficits(
        positions,
        direction,
        calc._rotor_diameters,
        calc._thrust_coefficients,
        model,
    )
    np.testing.assert_allclose(internal_matrix, ref_matrix, atol=1e-12)
    np.testing.assert_allclose(
        internal_total, np.sqrt(np.sum(ref_matrix ** 2, axis=0)), atol=1e-12
    )


@pytest.mark.parametrize("model", [JensenWake(0.07), GaussianWake(0.035)])
def test_aep_physically_sane(model):
    calc = _make_calculator(model)
    positions = np.array([
        [0.0, 0.0],
        [7.0 * 126.0, 0.0],
        [0.0, 7.0 * 126.0],
        [7.0 * 126.0, 7.0 * 126.0],
    ])
    result = calc.compute_farm_aep(positions)

    assert result.net_aep > 0.0
    assert result.gross_aep > result.net_aep
    # 尾流损失应在合理范围（此前错误符号下同轴下游风机近全停）
    assert 0.0 < result.wake_loss_pct < 30.0
    for tr in result.turbine_results:
        assert 0.0 <= tr.wake_loss_pct < 100.0
        assert tr.net_aep > 0.0


def test_wider_spacing_reduces_jensen_wake_loss():
    # 拉大同轴间距后尾流损失必须下降（回归核心缺陷）
    turbines = [create_default_turbine("V126-3.45MW") for _ in range(2)]
    calc = AEPCalculator(
        turbines=turbines,
        wind_resource=_single_direction_resource(270.0),
        wake_model=JensenWake(0.07),
        speed_step=1.0,
    )
    d = 126.0

    close = calc.compute_farm_aep(np.array([[0.0, 0.0], [5.0 * d, 0.0]]))
    wide = calc.compute_farm_aep(np.array([[0.0, 0.0], [20.0 * d, 0.0]]))
    assert close.wake_loss_pct > 0.0
    assert wide.wake_loss_pct < close.wake_loss_pct


def test_evaluate_layout_agrees_with_full_aep():
    calc = _make_calculator(JensenWake(0.07))
    positions = np.array([
        [0.0, 0.0],
        [7.0 * 126.0, 0.0],
        [0.0, 7.0 * 126.0],
        [7.0 * 126.0, 7.0 * 126.0],
    ])
    result = calc.compute_farm_aep(positions)
    quick = calc.evaluate_layout(positions)
    assert quick == pytest.approx(result.net_aep, rel=1e-9)
