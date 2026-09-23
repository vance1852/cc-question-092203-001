"""热力图与统一亏损契约的集成测试。"""

import numpy as np
import pytest

from wind_farm_opt.constraints.boundary import create_rectangular_boundary
from wind_farm_opt.core.wake import JensenWake, GaussianWake
from wind_farm_opt.visualization.plotting import plot_wake_heatmap


@pytest.mark.parametrize("model", [JensenWake(0.07), GaussianWake(0.035)])
def test_heatmap_renders_with_finite_bounded_deficit(model, tmp_path):
    boundary = create_rectangular_boundary(2000, 2000)
    d = 126.0
    positions = np.array([
        [-400.0, 0.0],
        [7.0 * d - 400.0, 0.0],
        [0.0, 500.0],
    ])
    diameters = np.full(3, d)
    cts = np.full(3, 0.82)

    out = tmp_path / "heatmap.png"
    plot_wake_heatmap(
        positions=positions,
        boundary=boundary,
        wake_model=model,
        wind_direction=270.0,
        rotor_diameters=diameters,
        thrust_coefficients=cts,
        grid_resolution=40,
        save_path=str(out),
        show=False,
    )
    assert out.exists() and out.stat().st_size > 0
