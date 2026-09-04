#!/usr/bin/env python3
"""Simulation-based evaluation of the MCL localization module.

Runs directly on the MCL implementation released with the robot code
(robinion_localization/scripts/mcl.py) and the same field configuration.

Experiment families:
  obs   -- observability: accuracy vs. number of visible landmarks (FOV sweep)
  imu   -- ablation: with / without IMU heading correction
  base  -- baseline comparison: least-squares trilateration vs. MCL
  odom  -- robustness: odometry noise sweep
  rate  -- robustness: filter update rate sweep
  kidnap-- robustness: recovery time after a kidnapped-robot event

Usage: python3 mcl_experiments.py <family> <seed> [extra]
Results are written as JSON to parts/<family>_<...>.json
"""
import sys, os, json, time
import numpy as np

REPO = os.environ.get('ROBINION_REPO', '/tmp/ros-robinion2')
sys.path.insert(0, os.path.join(REPO, 'robinion_localization/scripts'))
import mcl as _m
_m.random = np.random.random  # mcl.py calls random() without importing it
from mcl import (create_uniform_particles, predict, estimate, neff,
                 simple_resample, updateBasedOnDist, updateBasedOnOrientation)
from field import Field

FIELD_CFG = os.path.join(REPO, 'robinion_localization/config/mini_soccer_field.field')
FIELD = Field(FIELD_CFG)
LM = {k: np.array(v) for k, v in FIELD.landmarks.items()}
XR = (-FIELD.width / 2 * 0.9, FIELD.width / 2 * 0.9)
YR = (-FIELD.height / 2 * 0.9, FIELD.height / 2 * 0.9)

DEFAULTS = dict(N=5000, sensor_std=0.1, fov_deg=69.0, rate=20.0,
                motion_std=(0.05, 0.05), imu=True, imu_noise=0.05,
                r_theta=0.1, steps=300)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def visible(x, fov):
    """Landmarks inside the camera's horizontal field of view."""
    out = {}
    for k, p in LM.items():
        b = wrap(np.arctan2(p[1] - x[1], p[0] - x[0]) - x[2])
        if abs(b) <= fov / 2:
            out[k] = p
    return out


def trilaterate(z, guess):
    """Least-squares position estimate from range measurements (baseline)."""
    if len(z) < 3:
        return None
    keys = list(z.keys())
    P = np.array([LM[k] for k in keys])
    d = np.array([z[k] for k in keys])
    p = guess.copy()
    for _ in range(20):  # Gauss-Newton
        diff = p - P
        r = np.linalg.norm(diff, axis=1)
        r = np.maximum(r, 1e-6)
        J = diff / r[:, None]
        res = r - d
        try:
            step = np.linalg.lstsq(J, -res, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        p = p + step
        if np.linalg.norm(step) < 1e-6:
            break
    return p


def run_trial(seed, **kw):
    cfg = dict(DEFAULTS)
    cfg.update(kw)
    N = cfg['N']
    fov = np.radians(cfg['fov_deg'])
    dt = 1.0 / cfg['rate']
    steps = int(cfg['steps'] * cfg['rate'] / 20.0)  # keep wall-clock duration fixed

    rng = np.random.default_rng(seed)
    x = np.array([rng.uniform(*XR), rng.uniform(*YR), rng.uniform(0, 2 * np.pi)])
    particles = create_uniform_particles((-FIELD.width / 2, FIELD.width / 2),
                                         (-FIELD.height / 2, FIELD.height / 2),
                                         (0, 2 * np.pi), N)
    weights = np.ones(N) / N

    errs, herrs, nvis, times, base_errs = [], [], [], [], []
    kidnap_at = cfg.get('kidnap_at')

    for t in range(steps):
        if kidnap_at is not None and t == int(kidnap_at * cfg['rate']):
            # teleport the robot without telling the filter
            x[0], x[1] = rng.uniform(*XR), rng.uniform(*YR)
            x[2] = rng.uniform(0, 2 * np.pi)

        w = rng.uniform(-0.5, 0.5) / 8.0
        v = rng.uniform(0, 0.5)
        if not (XR[0] < x[0] < XR[1] and YR[0] < x[1] < YR[1]):
            w = wrap(np.arctan2(-x[1], -x[0]) - x[2]) * 0.5
        x[2] = (x[2] + w) % (2 * np.pi)
        x[0] += np.cos(x[2]) * v * dt
        x[1] += np.sin(x[2]) * v * dt

        vis = visible(x, fov)
        z = {k: float(np.linalg.norm(x[:2] - p) + rng.normal(0, cfg['sensor_std']))
             for k, p in vis.items()}
        nvis.append(len(z))

        t0 = time.perf_counter()
        predict(particles, [v, 0.0, w], list(cfg['motion_std']), dt)
        if z:
            updateBasedOnDist(particles, weights, z, cfg['sensor_std'], FIELD)
        if cfg['imu']:
            z_ori = (x[2] + rng.normal(0, cfg['imu_noise'])) % (2 * np.pi)
            updateBasedOnOrientation(particles, weights, z_ori, cfg['r_theta'])
        if neff(weights) < N / 2:
            simple_resample(particles, weights)
        mean, _ = estimate(particles, weights)
        times.append(time.perf_counter() - t0)

        errs.append(float(np.linalg.norm(mean[:2] - x[:2])))
        herrs.append(abs(float(wrap(mean[2] - x[2]))))

        if cfg.get('baseline'):
            p = trilaterate(z, mean[:2])
            base_errs.append(float(np.linalg.norm(p - x[:2])) if p is not None else np.nan)

    out = dict(seed=seed, cfg={k: (list(v) if isinstance(v, tuple) else v)
                               for k, v in cfg.items()},
               errs=errs, herrs=herrs, nvis=nvis,
               ms=float(np.mean(times) * 1000))
    if base_errs:
        out['base_errs'] = base_errs
    return out


def steady_rmse(e, skip_s=5.0, rate=20.0):
    a = np.array(e, dtype=float)
    a = a[int(skip_s * rate):]
    a = a[~np.isnan(a)]
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else float('nan')


def save(family, tag, res):
    os.makedirs('parts', exist_ok=True)
    fn = f"parts/{family}_{tag}.json"
    json.dump(res, open(fn, 'w'))
    print("saved", fn)


if __name__ == '__main__':
    family = sys.argv[1]
    seed = int(sys.argv[2])
    extra = sys.argv[3] if len(sys.argv) > 3 else None

    if family == 'obs':          # extra = fov in degrees
        fov = float(extra)
        save(family, f"{fov}_{seed}", run_trial(seed, fov_deg=fov))
    elif family == 'imu':        # extra = 'on' | 'off'
        save(family, f"{extra}_{seed}", run_trial(seed, imu=(extra == 'on')))
    elif family == 'base':
        save(family, f"{seed}", run_trial(seed, baseline=True))
    elif family == 'odom':       # extra = odometry noise std
        s = float(extra)
        save(family, f"{s}_{seed}", run_trial(seed, motion_std=(s, s)))
    elif family == 'rate':       # extra = update rate in Hz
        r = float(extra)
        save(family, f"{r}_{seed}", run_trial(seed, rate=r))
    elif family == 'kidnap':
        save(family, f"{seed}", run_trial(seed, kidnap_at=7.5, steps=300))
    else:
        raise SystemExit(f"unknown family: {family}")
