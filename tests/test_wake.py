"""尾流模型与 AEP/热力图一致性的回归测试。

这些测试锁定修复后的模型契约：
- Jensen 中心线亏损 = (1-sqrt(1-Ct))*(D/(D+2kx))^2，随下游距离单调衰减；
- Gaussian 既有数值行为（Bastankhah-Porté-Agel）保持不变；
- 单对交互、全场叠加、热力图共用同一套几何与叠加规则；
- 零距离、非物理推力系数、标量/数组输入均有确定行为。
"""

import numpy as np
import pytest

from wind_farm_opt.constraints.boundary import create_rectangular_boundary
from wind_farm_opt.core.turbine import create_default_turbine
from wind_farm_opt.core.wake import (
    GaussianWake,
    JensenWake,
    compute_wake_interactions,
    superpose_wakes,
    wake_deficit_field,
    wake_deficit_from_turbine,
    wind_unit_vector,
)
from wind_farm_opt.core.wind_resource import WindResource, WindSector
from wind_farm_opt.farm.aep import AEPCalculator
from wind_farm_opt.visualization.plotting import plot_wake_heatmap

D = 126.0
CT = 0.82


# ---------------------------------------------------------------------------
# Jensen 模型
# ---------------------------------------------------------------------------

class TestJensen:
    def setup_method(self):
        self.model = JensenWake(wake_decay=0.07)

    def test_scalar_and_array_return_types(self):
        scalar = self.model.velocity_deficit(5.0 * D, D, CT)
        assert isinstance(scalar, float)

        arr = self.model.velocity_deficit(np.array([5.0, 10.0]) * D, D, CT)
        assert isinstance(arr, np.ndarray) and arr.shape == (2,)

        assert isinstance(self.model.wake_radius(5.0 * D, D), float)
        assert isinstance(self.model.radial_profile(1.0, 100.0), float)
        r = self.model.radial_profile(np.array([1.0, 200.0]), 100.0)
        assert isinstance(r, np.ndarray)

    @pytest.mark.parametrize("distance", [0.0, -D, -10.0 * D])
    def test_non_downstream_distance_is_zero(self, distance):
        assert self.model.velocity_deficit(distance, D, CT) == 0.0

    def test_rotor_plane_value(self):
        # x -> 0+ 时亏损趋于感应亏损 1-sqrt(1-Ct)
        near = self.model.velocity_deficit(1e-9, D, CT)
        assert near == pytest.approx(1.0 - np.sqrt(1.0 - CT), abs=1e-6)

    def test_golden_values(self):
        xs = np.array([5.0, 10.0, 20.0]) * D
        deficits = self.model.velocity_deficit(xs, D, CT)
        expected_factor = 1.0 - np.sqrt(1.0 - CT)
        for deficit, x in zip(deficits, xs):
            ratio = D / (D + 2.0 * 0.07 * x)
            assert deficit == pytest.approx(expected_factor * ratio ** 2, rel=1e-12)

    def test_monotonic_decay_and_far_field_recovery(self):
        xs = np.concatenate([[1e-6], np.linspace(0.5, 100.0, 500)]) * D
        deficits = self.model.velocity_deficit(xs, D, CT)
        # 有效远场亏损必须严格单调减小
        assert np.all(np.diff(deficits) < 0.0)
        # 远场接近自由来流，而不是接近完全停机
        assert deficits[-1] < 0.005
        assert np.all(deficits < 1.0)

    def test_zero_thrust_gives_no_deficit(self):
        xs = np.array([1.0, 5.0, 20.0]) * D
        assert np.all(self.model.velocity_deficit(xs, D, 0.0) == 0.0)

    @pytest.mark.parametrize("bad_ct", [-0.5, 1.5, 2.0])
    def test_nonphysical_thrust_is_clipped(self, bad_ct):
        x = 5.0 * D
        clipped = self.model.velocity_deficit(x, D, bad_ct)
        boundary_ct = 0.0 if bad_ct < 0.0 else 1.0
        assert clipped == pytest.approx(
            self.model.velocity_deficit(x, D, boundary_ct)
        )
        assert 0.0 <= clipped <= 1.0

    def test_unit_thrust_is_finite(self):
        deficits = self.model.velocity_deficit(np.array([0.1, 5.0, 50.0]) * D, D, 1.0)
        assert np.all(np.isfinite(deficits))

    def test_nan_thrust_raises(self):
        with pytest.raises(ValueError):
            self.model.velocity_deficit(5.0 * D, D, np.nan)

    def test_invalid_rotor_diameter_raises(self):
        with pytest.raises(ValueError):
            self.model.velocity_deficit(5.0 * D, 0.0, CT)

    def test_wake_radius_expands_linearly(self):
        radii = self.model.wake_radius(np.array([0.0, 100.0, 1000.0]), D)
        assert radii[0] == pytest.approx(D / 2.0)
        assert radii[1] == pytest.approx(D / 2.0 + 0.07 * 100.0)
        assert np.all(np.diff(radii) > 0.0)

    def test_top_hat_radial_profile(self):
        profile = self.model.radial_profile(np.array([0.0, 50.0, 100.0, 100.1]), 100.0)
        np.testing.assert_allclose(profile, [1.0, 1.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Gaussian 模型：锁定修复前在合法 Ct 下的既有数值
# ---------------------------------------------------------------------------

class TestGaussian:
    def setup_method(self):
        self.model = GaussianWake(wake_decay=0.035, near_wake_length=3.0)

    def test_golden_centerline_deficits(self):
        xs = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 30.0]) * D
        deficits = self.model.velocity_deficit(xs, D, 0.8)
        expected = [0.0, 0.227405, 0.227405, 0.227405,
                    0.169263, 0.095073, 0.061194, 0.024305]
        np.testing.assert_allclose(deficits, expected, atol=1e-6)

    def test_golden_wake_radius(self):
        xs = np.array([0.0, 1.0, 3.0, 5.0, 10.0, 30.0]) * D
        radii = self.model.wake_radius(xs, D)
        expected = [63.0, 89.46, 89.46, 107.1, 151.2, 327.6]
        np.testing.assert_allclose(radii, expected, atol=1e-6)

    def test_far_field_monotonic_and_recovering(self):
        xs = np.linspace(3.0, 100.0, 200) * D
        deficits = self.model.velocity_deficit(xs, D, 0.8)
        assert np.all(np.diff(deficits) < 0.0)
        assert deficits[-1] < 0.01

    def test_non_downstream_distance_is_zero(self):
        assert self.model.velocity_deficit(0.0, D, 0.8) == 0.0
        assert self.model.velocity_deficit(-5.0 * D, D, 0.8) == 0.0

    def test_unit_thrust_is_finite(self):
        deficits = self.model.velocity_deficit(np.array([3.0, 10.0]) * D, D, 1.0)
        assert np.all(np.isfinite(deficits))

    def test_nonphysical_thrust_is_clipped(self):
        x = 10.0 * D
        assert self.model.velocity_deficit(x, D, -1.0) == pytest.approx(0.0)
        assert self.model.velocity_deficit(x, D, 2.0) == pytest.approx(
            self.model.velocity_deficit(x, D, 1.0)
        )

    def test_radial_profile(self):
        radius = 200.0
        center = self.model.radial_profile(0.0, radius)
        assert center == pytest.approx(1.0)
        edge = self.model.radial_profile(radius / 2.0, radius)
        assert edge == pytest.approx(np.exp(-0.5))
        assert 0.0 < self.model.radial_profile(500.0, radius) < 1.0

    def test_radial_profile_array_and_degenerate_radius(self):
        prof = self.model.radial_profile(np.array([0.0, 100.0]), 200.0)
        assert prof.shape == (2,)
        assert self.model.radial_profile(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# 叠加
# ---------------------------------------------------------------------------

class TestSuperposition:
    def test_sum_of_squares(self):
        total = superpose_wakes(np.array([0.3, 0.4]))
        assert total == pytest.approx(0.5)

    def test_linear(self):
        total = superpose_wakes(np.array([0.3, 0.4]), method="linear")
        assert total == pytest.approx(0.7)

    def test_clipped_to_one(self):
        assert superpose_wakes(np.array([0.9, 0.9])) <= 1.0
        assert superpose_wakes(np.array([0.9, 0.9]), method="linear") == pytest.approx(1.0)

    def test_empty_input(self):
        total = superpose_wakes(np.zeros((0, 3)))
        np.testing.assert_array_equal(total, np.zeros(3))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            superpose_wakes(np.array([0.1]), method="banana")

    def test_nonfinite_raises(self):
        with pytest.raises(ValueError):
            superpose_wakes(np.array([0.1, np.nan]))


# ---------------------------------------------------------------------------
# 统一几何契约：wake_deficit_from_turbine
# ---------------------------------------------------------------------------

class TestDeficitFieldContract:
    def setup_method(self):
        self.jensen = JensenWake(0.07)
        self.wind = wind_unit_vector(270.0)  # 西风，流向 +x

    def test_aligned_downstream_matches_centerline(self):
        x = 7.0 * D
        point_deficit = wake_deficit_from_turbine(
            self.jensen, np.zeros(2), np.array([x, 0.0]), self.wind, D, CT
        )
        direct = self.jensen.velocity_deficit(x, D, CT)
        assert point_deficit == pytest.approx(direct)

    def test_upstream_point_is_zero(self):
        deficit = wake_deficit_from_turbine(
            self.jensen, np.zeros(2), np.array([-5.0 * D, 0.0]), self.wind, D, CT
        )
        assert deficit == 0.0

    def test_coincident_point_is_zero(self):
        deficit = wake_deficit_from_turbine(
            self.jensen, np.zeros(2), np.zeros(2), self.wind, D, CT
        )
        assert deficit == 0.0

    def test_outside_top_hat_is_zero(self):
        # x=7D 处 Jensen 尾流半径 = D/2 + 0.07*7D ≈ 0.99D；横向 2D 必在外部
        deficit = wake_deficit_from_turbine(
            self.jensen, np.zeros(2), np.array([7.0 * D, 2.0 * D]),
            self.wind, D, CT,
        )
        assert deficit == 0.0

    def test_grid_shape_and_bounds(self):
        points = np.random.default_rng(0).uniform(-500, 2000, size=(8, 6, 2))
        field = wake_deficit_from_turbine(
            self.jensen, np.zeros(2), points, self.wind, D, CT
        )
        assert field.shape == (8, 6)
        assert np.all((field >= 0.0) & (field <= 1.0))


# ---------------------------------------------------------------------------
# 三个消费者（交互明细 / AEP 全场 / 热力图）方向与量级一致
# ---------------------------------------------------------------------------

def _unidirectional_resource(speed=8.5):
    c = speed / 0.8862269  # Gamma(1.5)，k=2 时均值 c*Gamma(1+1/k)
    sector = WindSector(
        direction_center=270.0,
        direction_width=360.0,
        frequency=1.0,
        mean_speed=speed,
        weibull_k=2.0,
        weibull_c=c,
    )
    return WindResource([sector])


def _aligned_layout(spacing):
    return np.array([[0.0, 0.0], [spacing, 0.0]])


@pytest.mark.parametrize(
    "model", [JensenWake(0.07), GaussianWake(0.035)], ids=["jensen", "gaussian"]
)
@pytest.mark.parametrize("spacing_d", [3.0, 7.0, 15.0])
def test_interaction_aep_heatmap_agree(model, spacing_d, tmp_path):
    spacing = spacing_d * D
    positions = _aligned_layout(spacing)
    turbines = [create_default_turbine("V126-3.45MW") for _ in range(2)]
    diameters = np.array([t.rotor_diameter for t in turbines])
    cts = np.array([t.thrust_coefficient for t in turbines])
    wind = wind_unit_vector(270.0)

    # 交互明细
    interactions = compute_wake_interactions(
        positions, 270.0, diameters, cts, model,
        [t.power_curve for t in turbines], free_stream_speed=8.5,
    )
    pair = next(it for it in interactions
                if it.upstream_idx == 0 and it.downstream_idx == 1)

    # 统一契约函数（AEP 与热力图内部都走这里）
    contract = wake_deficit_from_turbine(
        model, positions[0], positions[1], wind, diameters[0], cts[0]
    )

    # AEP 全场叠加
    calc = AEPCalculator(
        turbines=turbines,
        wind_resource=_unidirectional_resource(),
        wake_model=model,
        speed_step=1.0,
    )
    field = calc._compute_wake_deficit_field(positions, 270.0)

    assert pair.velocity_deficit == pytest.approx(contract)
    assert field[1] == pytest.approx(contract)
    # 首台风机无上游，亏损必须为 0
    assert field[0] == 0.0

    # 热力图所用的亏损场：在机位处采样必须与交互明细一致，
    # 在首台风机上游侧必须为 0
    shifted = positions + np.array([3.0 * D, 2.0 * D])
    sample_points = np.vstack([
        shifted,
        shifted[0] + np.array([-2.0 * D, 0.0]),
    ])
    grid_field = wake_deficit_field(
        model, shifted, sample_points, 270.0, diameters, cts
    )
    assert grid_field[1] == pytest.approx(contract)
    assert grid_field[0] == 0.0
    assert grid_field[2] == 0.0

    boundary = create_rectangular_boundary(
        max(spacing + 4.0 * D, 1000.0), 4.0 * D
    )
    save_path = tmp_path / "heatmap.png"
    plot_wake_heatmap(
        shifted, boundary, model, 270.0,
        diameters, cts, grid_resolution=80,
        save_path=str(save_path), show=False,
    )
    assert save_path.exists()


@pytest.mark.parametrize(
    "model", [JensenWake(0.07), GaussianWake(0.035)], ids=["jensen", "gaussian"]
)
def test_farther_spacing_recovers_power(model):
    turbines = [create_default_turbine("V126-3.45MW") for _ in range(2)]
    calc = AEPCalculator(
        turbines=turbines,
        wind_resource=_unidirectional_resource(),
        wake_model=model,
        speed_step=0.5,
    )
    close = calc.evaluate_layout(_aligned_layout(3.0 * D))
    far = calc.evaluate_layout(_aligned_layout(15.0 * D))
    # 拉大同轴间距后亏损必须减小、净 AEP 必须升高
    assert far > close


def test_jensen_aep_loss_reasonable_magnitude():
    turbines = [create_default_turbine("V126-3.45MW") for _ in range(2)]
    model = JensenWake(0.07)
    calc = AEPCalculator(
        turbines=turbines,
        wind_resource=_unidirectional_resource(),
        wake_model=model,
        speed_step=0.5,
    )
    result = calc.compute_farm_aep(_aligned_layout(7.0 * D))
    # 修复前远场近停机会给出荒谬的高损失；7D 同轴损失应在可信区间
    assert 0.0 < result.wake_loss_pct < 25.0
    assert result.net_aep > 0.0


@pytest.mark.parametrize(
    "model", [JensenWake(0.07), GaussianWake(0.035)], ids=["jensen", "gaussian"]
)
def test_multi_turbine_chain_direction_consistency(model):
    # 三机同轴串列：沿下游方向，累计亏损只作用在下游机上且量级递增合理
    turbines = [create_default_turbine("V126-3.45MW") for _ in range(3)]
    diameters = np.array([t.rotor_diameter for t in turbines])
    cts = np.array([t.thrust_coefficient for t in turbines])
    positions = np.array([[0.0, 0.0], [7.0 * D, 0.0], [14.0 * D, 0.0]])
    field = wake_deficit_field(
        model, positions, positions, 270.0, diameters, cts
    )
    assert field[0] == 0.0
    assert 0.0 < field[1] < field[2] < 1.0
    # 两台上游叠加的平方和亏损不应超过最大单机贡献的 sqrt(2) 倍
    pair_max = max(
        wake_deficit_from_turbine(
            model, positions[s], positions[2], wind_unit_vector(270.0),
            diameters[s], cts[s],
        )
        for s in (0, 1)
    )
    assert field[2] <= pair_max * np.sqrt(2.0) + 1e-12
