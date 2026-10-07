"""Regression checks for exact nuisance-split minimization in the live certificate."""
import os
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
import certs


def test_stratified_minimum_includes_nongrid_endpoint():
    # Increasing, concave Qs. At the first crossing all the mixture mean is
    # allocated to stratum 1; its feasible endpoint is not a fixed-grid point.
    a = np.array([-0.1]); d = np.array([-0.04])
    q = (0.413, 0.587); C = 0.2
    stats = [(a, np.array([1.0]), d), (a, np.array([2.0]), d)]
    z = np.zeros((1, 64))
    expected_m1 = (1.0 - np.sqrt(1.0 - 0.4 * (C + 0.08))) / 0.2
    expected_bound = q[0] * expected_m1
    for n_split in (3, 2001):
        with patch.object(certs, "bet_quadratic", side_effect=stats):
            bound = certs.exc_bet_stratified([(z, z), (z, z)], q, 1.0, C,
                                            n_split=n_split, n_bisect=55)
        np.testing.assert_allclose(bound, expected_bound, rtol=0, atol=1e-14)


def test_stratified_bound_is_invariant_to_stratum_order():
    rng = np.random.default_rng(904)
    strata = []
    for n in (401, 699):
        w = 3.0 * (rng.random((4, n)) < 1.0 / 3.0)
        e = (rng.random((4, n)) < 0.1).astype(float)
        strata.append((w, e))
    args = dict(W=3.0, C=np.log(4 / 0.05))
    forward = certs.exc_bet_stratified(strata, (0.413, 0.587), **args)
    reverse = certs.exc_bet_stratified(strata[::-1], (0.587, 0.413), **args)
    np.testing.assert_allclose(forward, reverse, rtol=0, atol=1e-13)


if __name__ == "__main__":
    test_stratified_minimum_includes_nongrid_endpoint()
    test_stratified_bound_is_invariant_to_stratum_order()
    print("stratified certificate tests passed")
