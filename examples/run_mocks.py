"""
run_mocks.py -- driver for make_mocks.py.

Examples
--------
# one-time setup (downloads ~0.7 GB of PHOENIX R=10000 nodes, builds a 0.25 GB cache)
python run_mocks.py setup

# (A) base set: N=80000, no RV jitter, fixed R=1800
python run_mocks.py make --name base --n 80000 --seed 1

# (B) jitter sets: residual RV offsets and/or variable LSF
python run_mocks.py make --name jitter_v20_Rvar --n 5000 --sigma-v 20 --R-mode uniform --seed 2
python run_mocks.py make --name jitter_v50      --n 20000 --sigma-v 50 --seed 3
#   (loop over --sigma-v 5 20 50 100; add --R-mode uniform --R-min 1600 --R-max 2000 for LSF variation)

# Other knobs: --snr-min/--snr-max/--snr-power, --read-noise, --cont-method,
# --cont-window, --f-giant, --feh-mu, --feh-sig.  Run with -h for the list.

Outputs go to mock/data/<name>.h5 (see make_mocks.generate_dataset for the layout).
Using the same seed and population/S/N settings gives the *same stars and S/N*
in two sets, so e.g. base and a jitter variant differ only by the nuisance knob
(the noise realization is also identical because it is seeded per chunk).
"""
import argparse
import os
import time

import make_mocks as mm

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["setup", "make"])
    p.add_argument("--name", default="base")
    p.add_argument("--n", type=int, default=80000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--sigma-v", type=float, default=0.0)
    p.add_argument("--R-mode", choices=["fixed", "uniform"], default="fixed")
    p.add_argument("--R-fixed", type=float, default=1800.0)
    p.add_argument("--R-min", type=float, default=1600.0)
    p.add_argument("--R-max", type=float, default=2000.0)
    p.add_argument("--snr-min", type=float, default=30.0)
    p.add_argument("--snr-max", type=float, default=400.0)
    p.add_argument("--snr-power", type=float, default=2.5)
    p.add_argument("--read-noise", type=float, default=8.0)
    p.add_argument("--cont-method", default="clipped_median",
                   choices=["median", "clipped_median", "mean", "percentile"])
    p.add_argument("--cont-window", type=int, default=41)
    p.add_argument("--f-giant", type=float, default=0.25)
    p.add_argument("--feh-mu", type=float, default=-0.25)
    p.add_argument("--feh-sig", type=float, default=0.30)
    p.add_argument("--chunk", type=int, default=1000)
    p.add_argument("--outdir", default=os.path.join(HERE, "data"))
    a = p.parse_args()

    if a.cmd == "setup":
        t = time.time()
        res, missing = mm.fetch_phoenix_nodes()
        print("download:", res, "missing:", missing, f"{time.time() - t:.0f} s")
        mm.preprocess_nodes(mm.Grid())
        print("cache written:", mm.CACHE_FILE, f"{time.time() - t:.0f} s")
        return

    cfg = mm.MockConfig(
        n_star=a.n, seed=a.seed, sigma_v=a.sigma_v, R_mode=a.R_mode, R_fixed=a.R_fixed,
        R_range=(a.R_min, a.R_max), chunk=a.chunk,
        population=mm.PopulationConfig(f_giant=a.f_giant, feh_mu=a.feh_mu, feh_sig=a.feh_sig),
        obs=mm.ObsConfig(snr_min=a.snr_min, snr_max=a.snr_max, snr_power=a.snr_power,
                         read_noise=a.read_noise),
        continuum=mm.ContinuumConfig(method=a.cont_method, window=a.cont_window),
    )
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, a.name + ".h5")
    t = time.time()
    mm.generate_dataset(out, cfg)
    print(f"wrote {out} in {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
