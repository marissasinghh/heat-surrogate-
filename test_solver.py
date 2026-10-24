import numpy as np
from solver import ftcs_step

def test_boundaries_stay_zero():
    rng = np.random.default_rng(0)
    u = rng.uniform(-1, 1, 20)
    u[0], u[-1] = 0.0, 0.0
    u_new = ftcs_step(u, 0.4)
    assert u_new[0] == 0.0 and u_new[-1] == 0.0
    assert u_new.shape == u.shape

def test_hand_example():
    u = np.array([0.0, 1.0, 0.0])
    u_new = ftcs_step(u, 0.25)
    new_middle = 0.5
    assert np.isclose(u_new[1], new_middle)

def test_zero_curvature():
    u = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
    u_new =ftcs_step(u, 0.25)
    assert np.isclose(u_new[2], u[2])

def test_input_not_modified():
    u = np.arange(7, dtype=float)    
    u_before = u.copy()
    u_new = ftcs_step(u, 0.4)
    assert np.array_equal(u, u_before)

if __name__ == "__main__":
    test_boundaries_stay_zero()
    test_hand_example()
    test_zero_curvature()
    test_input_not_modified()
    print("All tests passed!")