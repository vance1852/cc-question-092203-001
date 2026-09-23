"""尾流模型与多路径一致性的回归测试。

锁定修复内容：
- Jensen 中心线亏损定义为 ``1 - u/U0`` 且随下游距离单调衰减；
- 零距离、非物理推力系数、标量/数组输入等边界行为；
- Gaussian 与径向剖面的既有数值行为不被破坏；
- 单对交互、全场 AEP 叠加与热力图网格共用同一份模型契约。
"""

import numpy as np
import pytest

from wind_farm_opt.core.wake import (
    JensenWake,
    GaussianWake,
    superpose_wakes,
    wind_unit_vector,
    wake_geometry,
    point_velocity_deficit,
    compute_pairwise_deficits,
    compute_deficit_field,
    compute_wake_interactions,
)

D0 = 126.0
CT = 0.82


# ---------------------------------------------------------------------------
# Jensen 亏损定义与衰减行为
# ---------------------------------------------------------------------------

class TestJensenDeficit:
    def setup_method(self):
        self.model = JensenWake(wake_decay=0.07)

    def test_matches_analytical_formula(self):
        x = 5.0 * D0
        expected = (1.0 - np.sqrt(1.0 - CT)) / (1.0 + 2.0 * 0.07 * 5.0) ** 2
        assert self.model.velocity_deficit(x, D0, CT) == pytest.approx(expected)

    def test_rotor_plane_value_is_physically_bounded(self):
        # x -> 0+ 时亏损趋近 1-sqrt(1-Ct)，且严格小于 1
        near = self.model.velocity_deficit(1e-6 * D0, D0, CT)
        assert near == pytest.approx(1.0 - np.sqrt(1.0 - CT), abs=1e-4)
        assert near < 1.0

    def test_far_field_decays_to_zero(self):
        far = self.model.velocity_deficit(200.0 * D0, D0, CT)
        assert far == pytest.approx(0.0, abs=1e-3)

    def test_monotonically_decreasing(self):
        x = np.linspace(0.5, 50.0, 200) * D0
        deficit = self.model.velocity_deficit(x, D0, CT)
        assert np.all(np.diff(deficit) < 0.0)

    def test_far_field_smaller_than_near_field(self):
        # 回归核心缺陷：此前拉大距离后亏损反而接近 1（近全停机）
        near = self.model.velocity_deficit(2.0 * D0, D0, CT)
        far = self.model.velocity_deficit(30.0 * D0, D0, CT)
        assert far < near
        assert far < 0.05

    def test_zero_and_negative_distance(self):
        assert self.model.velocity_deficit(0.0, D0, CT) == 0.0
        assert self.model.velocity_deficit(-10.0 * D0, D0, CT) == 0.0
        arr = self.model.velocity_deficit(np.array([0.0, -1.0, 5.0]) * D0, D0, CT)
        assert arr[0] == 0.0
        assert arr[1] == 0.0
        assert arr[2] > 0.0

    def test_nonphysical_thrust_coefficient_clipped(self):
        x = 5.0 * D0
        assert self.model.velocity_deficit(x, D0, 1.5) == pytest.approx(
            self.model.velocity_deficit(x, D0, 1.0)
        )
        assert self.model.velocity_deficit(x, D0, -0.5) == 0.0

    def test_scalar_vs_array(self):
        x = 5.0 * D0
        scalar = self.model.velocity_deficit(x, D0, CT)
        assert isinstance(scalar, float)
        array = self.model.velocity_deficit(np.array([x, x]), D0, CT)
        assert isinstance(array, np.ndarray)
        np.testing.assert_allclose(array, [scalar, scalar])

    def test_broadcast_diameter_and_ct_arrays(self):
        x = np.array([5.0, 10.0]) * D0
        d = np.array([D0, D0])
        c = np.array([CT, CT])
        out = self.model.velocity_deficit(x, d, c)
        assert out.shape == (2,)
        assert np.all(np.isfinite(out))

    def test_no_nan_for_any_distance(self):
        x = np.linspace(-2.0, 100.0, 500) * D0
        out = self.model.velocity_deficit(x, D0, np.linspace(-0.5, 1.5, 500))
        assert np.all(np.isfinite(out))
        assert np.all((out >= 0.0) & (out <= 1.0))


# ---------------------------------------------------------------------------
# Gaussian：锁定既有数值行为，修复不得改变其物理结果
# ---------------------------------------------------------------------------

class TestGaussianRegression:
    def setup_method(self):
        self.model = GaussianWake(wake_decay=0.035)

    @pytest.mark.parametrize(
        "x_over_d, expected",
        [(1.0, 0.2419), (3.0, 0.2419), (5.0, 0.1789),
         (10.0, 0.0996), (20.0, 0.0444), (50.0, 0.0113)],
    )
    def test_reference_values_preserved(self, x_over_d, expected):
        assert self.model.velocity_deficit(x_over_d * D0, D0, CT) == pytest.approx(
            expected, abs=5e-4
        )

    def test_near_wake_plateau(self):
        # x < near_wake_length * D 时中心线亏损保持平台值
        v1 = self.model.velocity_deficit(1.0 * D0, D0, CT)
        v2 = self.model.velocity_deficit(2.5 * D0, D0, CT)
        assert v1 == pytest.approx(v2)

    def test_far_field_monotonic_and_decaying(self):
        x = np.linspace(3.0, 60.0, 200) * D0
        deficit = self.model.velocity_deficit(x, D0, CT)
        assert np.all(np.diff(deficit) < 0.0)
        assert deficit[-1] < deficit[0]

    def test_zero_distance_and_bounds(self):
        assert self.model.velocity_deficit(0.0, D0, CT) == 0.0
        out = self.model.velocity_deficit(np.array([0.0, 5.0]) * D0, D0, 1.0)
        assert np.all(np.isfinite(out))
        assert out[0] == 0.0
        assert 0.0 <= out[1] <= 1.0

    def test_ct_equal_one_does_not_nan(self):
        out = self.model.velocity_deficit(
            np.linspace(0.5, 10.0, 20) * D0, D0, 1.0
        )
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# 尾流半径与径向剖面
# ---------------------------------------------------------------------------

class TestRadiusAndRadialProfile:
    def test_jensen_radius_grows_linearly(self):
        model = JensenWake(0.07)
        r0 = model.wake_radius(0.0, D0)
        assert r0 == pytest.approx(D0 / 2.0)
        r10 = model.wake_radius(10.0 * D0, D0)
        assert r10 == pytest.approx(D0 / 2.0 + 0.07 * 10.0 * D0)

    def test_jensen_tophat_profile(self):
        model = JensenWake(0.07)
        wr = model.wake_radius(5.0 * D0, D0)
        assert model.radial_profile(0.0, wr) == 1.0
        assert model.radial_profile(wr - 1.0, wr) == 1.0
        assert model.radial_profile(wr + 1.0, wr) == 0.0

    def test_gaussian_profile_center_and_tail(self):
        model = GaussianWake(0.035)
        wr = model.wake_radius(10.0 * D0, D0)
        assert model.radial_profile(0.0, wr) == pytest.approx(1.0)
        tail = model.radial_profile(wr, wr)
        assert tail == pytest.approx(np.exp(-2.0))
        far = model.radial_profile(5.0 * wr, wr)
        assert far < 1e-6

    def test_radial_profile_array_broadcast(self):
        model = JensenWake(0.07)
        r = np.array([0.0, 50.0, 500.0])
        out = model.radial_profile(r, np.array([100.0, 100.0, 100.0]))
        np.testing.assert_allclose(out, [1.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# 叠加
# ---------------------------------------------------------------------------

class TestSuperposition:
    def test_sum_of_squares(self):
        total = superpose_wakes(np.array([0.3, 0.4]))
        assert total == pytest.approx(0.5)

    def test_linear(self):
        assert superpose_wakes(np.array([0.3, 0.4]), method="linear") == pytest.approx(0.7)

    def test_clipped_to_one(self):
        assert superpose_wakes(np.array([0.9, 0.9])) <= 1.0
        assert superpose_wakes(np.array([0.9, 0.9]), method="linear") == pytest.approx(1.0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            superpose_wakes(np.array([0.1]), method="bogus")

    def test_no_nan(self):
        out = superpose_wakes(np.array([np.nan, 0.2]))
        assert np.isfinite(out)


# ---------------------------------------------------------------------------
# 几何契约与三路径一致性
# ---------------------------------------------------------------------------

def test_wind_unit_vector_direction_convention():
    # 270° = 西风，风向东吹 (+x)
    v = wind_unit_vector(270.0)
    np.testing.assert_allclose(v, [1.0, 0.0], atol=1e-12)
    # 0° = 北风，风向南吹 (-y)
    v = wind_unit_vector(0.0)
    np.testing.assert_allclose(v, [0.0, -1.0], atol=1e-12)


def test_wake_geometry_centerline_and_upstream():
    # 西风向下，受影点在风机正东 5D：轴向 5D、横向 0
    delta = np.array([5.0 * D0, 0.0])
    xd, cr, mask = wake_geometry(delta, 270.0)
    assert mask
    assert xd == pytest.approx(5.0 * D0)
    assert cr == pytest.approx(0.0)

    # 上游（正西）不构成尾流
    xd, cr, mask = wake_geometry(np.array([-5.0 * D0, 0.0]), 270.0)
    assert not mask
    assert xd == 0.0 and cr == 0.0

    # 重合点
    xd, cr, mask = wake_geometry(np.array([0.0, 0.0]), 270.0)
    assert not mask


def _coaxial_positions(spacing_over_d):
    return np.array([
        [0.0, 0.0],
        [spacing_over_d * D0, 0.0],
    ])


@pytest.mark.parametrize("model", [JensenWake(0.07), GaussianWake(0.035)])
class TestCrossPathConsistency:
    """同一布局：交互明细、两两矩阵/叠加、网格场必须给出同一亏损。"""

    def test_pairwise_matches_interactions(self, model):
        positions = _coaxial_positions(7.0)
        diameters = np.array([D0, D0])
        cts = np.array([CT, CT])

        matrix = compute_pairwise_deficits(positions, 270.0, diameters, cts, model)

        power_curve = np.column_stack([np.arange(0.0, 26.0), np.zeros(26)])
        interactions = compute_wake_interactions(
            positions, 270.0, diameters, cts, model,
            [power_curve, power_curve], free_stream_speed=8.0,
        )
        pair = next(
            it for it in interactions
            if it.upstream_idx == 0 and it.downstream_idx == 1
        )
        assert pair.velocity_deficit == pytest.approx(matrix[0, 1])
        assert matrix[1, 0] == 0.0  # 风机 1 在风机 0 的上游，无亏损

    def test_grid_field_matches_pairwise_at_turbine_location(self, model):
        positions = _coaxial_positions(7.0)
        diameters = np.array([D0, D0])
        cts = np.array([CT, CT])

        matrix = compute_pairwise_deficits(positions, 270.0, diameters, cts, model)
        field = compute_deficit_field(
            points=positions, positions=positions, wind_direction=270.0,
            rotor_diameters=diameters, thrust_coefficients=cts, wake_model=model,
        )
        # 风机 j 处的网格叠加亏损应等于列向平方和
        expected_total = np.sqrt(np.sum(matrix ** 2, axis=0))
        np.testing.assert_allclose(field, expected_total, atol=1e-12)

    def test_single_upstream_grid_equals_direct_deficit(self, model):
        positions = _coaxial_positions(7.0)
        point = np.array([[7.0 * D0, 0.0], [7.0 * D0, 3.0 * D0]])
        field = compute_deficit_field(
            points=point, positions=positions[:1], wind_direction=270.0,
            rotor_diameters=np.array([D0]),
            thrust_coefficients=np.array([CT]),
            wake_model=model,
        )
        direct_centerline = point_velocity_deficit(
            7.0 * D0, 0.0, D0, CT, model
        )
        direct_offaxis = point_velocity_deficit(
            7.0 * D0, 3.0 * D0, D0, CT, model
        )
        np.testing.assert_allclose(field, [direct_centerline, direct_offaxis], atol=1e-12)

    def test_coaxial_farther_spacing_reduces_deficit(self, model):
        diameters = np.array([D0, D0])
        cts = np.array([CT, CT])
        near = compute_pairwise_deficits(
            _coaxial_positions(5.0), 270.0, diameters, cts, model
        )[0, 1]
        far = compute_pairwise_deficits(
            _coaxial_positions(20.0), 270.0, diameters, cts, model
        )[0, 1]
        assert far < near


def test_three_paths_agree_on_multi_turbine_layout():
    """多风机、非同轴布局下三条路径的总亏损一致。"""
    model = JensenWake(0.07)
    rng = np.random.default_rng(7)
    positions = rng.uniform(-2000.0, 2000.0, size=(6, 2))
    diameters = np.full(6, D0)
    cts = np.full(6, CT)
    direction = 270.0

    matrix = compute_pairwise_deficits(positions, direction, diameters, cts, model)
    total = superpose_wakes(matrix)
    field = compute_deficit_field(
        points=positions, positions=positions, wind_direction=direction,
        rotor_diameters=diameters, thrust_coefficients=cts, wake_model=model,
    )
    np.testing.assert_allclose(field, total, atol=1e-12)

    # 每台风机处的亏损都应严格小于 1（不存在“近全停机”）
    assert np.all(total < 1.0)
    assert np.all(total >= 0.0)
