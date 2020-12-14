#!/usr/bin/env python3

import os
import argparse
import pickle
import yaml
import numpy as np
import pandas as pd
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.rc("text", usetex=True)
cmap_med = [
    "#F15A60",
    "#7AC36A",
    "#5A9BD4",
    "#FAA75B",
    "#9E67AB",
    "#CE7058",
    "#D77FB4",
    "#737373",
]
cmap = [
    "#EE2E2F",
    "#008C48",
    "#185AA9",
    "#F47D23",
    "#662C91",
    "#A21D21",
    "#B43894",
    "#010202",
]
dashseq = [
    (None, None),
    [10, 5],
    [10, 4, 3, 4],
    [3, 3],
    [10, 4, 3, 4, 3, 4],
    [3, 3],
    [3, 3],
]
markertype = ["s", "d", "o", "p", "h", "v"]


def parse_ic(fname):
    """Parse the Nalu yaml input file for the initial conditions"""
    with open(fname, "r") as stream:
        try:
            dat = yaml.full_load(stream)
            u0 = float(
                dat["realms"][0]["initial_conditions"][0]["value"]["velocity"][0]
            )
            rho0 = float(
                dat["realms"][0]["material_properties"]["specifications"][0]["value"]
            )
            mu = float(
                dat["realms"][0]["material_properties"]["specifications"][1]["value"]
            )
            dt = float(
                dat["Time_Integrators"][0]["StandardTimeIntegrator"]["time_step"]
            )
            turb_model = dat["realms"][0]["solution_options"]["turbulence_model"]

            return u0, rho0, mu, dt, turb_model

        except yaml.YAMLError as exc:
            print(exc)


# The Tukey window
# see https://en.wikipedia.org/wiki/Window_function#Tukey_window
def tukeyWindow(N, params={"alpha": 0.1}):
    alpha = params["alpha"]
    w = np.zeros(N)
    L = N + 1
    for n in np.arange(0, int(N // 2) + 1):
        if (0 <= n) and (n < 0.5 * alpha * L):
            w[n] = 0.5 * (1.0 - np.cos(2 * np.pi * n / (alpha * L)))
        elif (0.5 * alpha * L <= n) and (n <= N / 2):
            w[n] = 1.0
        else:
            print("Something wrong happened at n = ", n)
        if n != 0:
            w[N - n] = w[n]
    return w


# FFT's a signal, returns 1-sided frequency and spectra
def getFFT(t, y, normalize=False, window=True):
    n = len(y)
    k = np.arange(n)
    dt = np.mean(np.diff(t))
    frq = k / (n * dt)
    if window:
        w = tukeyWindow(n)
    else:
        w = 1.0
    if normalize:
        L = len(y)
    else:
        L = 1.0
    FFTy = np.fft.fft(w * y) / L

    # Take the one sided version of it
    freq = frq[range(int(n // 2))]
    FFTy = FFTy[range(int(n // 2))]
    return freq, FFTy


def get_spectra(y, dt, ref_timescale):
    n_size = np.size(y)
    n_sizeB2 = int(n_size / 2)
    yfft = 2.0 * fft.fft(y - np.average(y)) / n_size
    yfft_freq = fft.fftfreq(n_size, dt)

    return np.abs(yfft[:n_sizeB2]), yfft_freq[:n_sizeB2] * ref_timescale


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="A simple plot tool")
    parser.add_argument(
        "-f", "--fdir", nargs="+", help="Folders to plot", required=True
    )
    parser.add_argument(
        "--tmin", help="Min time for average (tau units)", type=float, default=0.0
    )
    parser.add_argument(
        "--tmax",
        help="Max time for average (tau units)",
        type=float,
        default=10000000.0,
    )
    args = parser.parse_args()

    # defaults
    rho = 1.225
    uInfty = 20.0
    cylDiameter = 6.0
    cylSpan = 4 * cylDiameter
    refArea = cylSpan * cylDiameter
    dynPres = 0.5 * rho * uInfty * uInfty
    refTime = cylDiameter / uInfty

    # Reference data
    refdir = "../../references/Cylinder/Data/"
    with open(os.path.join(refdir, "lit_data.pkl"), "rb") as f:
        expcases, expdata, explabels, cfdcases, cfddata, cfdlabels = pickle.load(f)
    fig = plt.figure("cd_re")
    for i, e in enumerate(expdata):
        p = plt.semilogx(
            expdata[e]["Reynolds_number"],
            expdata[e]["Cd"],
            color="gray",
            label=explabels[i],
        )
        p[0].set_dashes(dashseq[i])

    for i, c in enumerate(cfddata):
        plt.semilogx(
            cfddata[c]["Reynolds_number"],
            cfddata[c]["Cd"],
            color="gray",
            linestyle="",
            marker=markertype[i],
            label=cfdlabels[i],
        )

    refdir = "../../references/Cylinder/Data/Pressure/"
    with open(os.path.join(refdir, "lit_data.pkl"), "rb") as f:
        expcases, expdata, explabels, cfdcases, cfddata, cfdlabels = pickle.load(f)
    fig = plt.figure("cp")
    for i, e in enumerate(expdata):
        plt.plot(
            expdata[e]["Theta"],
            expdata[e]["Cp"],
            linestyle="--",
            lw=2,
            label=explabels[i],
        )
    for i, c in enumerate(cfddata):
        plt.plot(
            cfddata[c]["Theta"],
            cfddata[c]["Cp"],
            linestyle="--",
            lw=2,
            marker=markertype[i],
            label=cfdlabels[i],
        )

    for i, fdir in enumerate(args.fdir):
        yname = os.path.join(os.path.dirname(fdir), "cylinder.yaml")
        u0, rho0, mu, dt, turb_model = parse_ic(yname)
        model = turb_model.upper().replace("_", "-")
        forces = pd.read_csv(
            os.path.join(os.path.join(os.path.dirname(fdir), "forces.dat")),
            delim_whitespace=True,
        )
        forces.Time /= refTime
        forces["cd"] = (forces.Fpx + forces.Fvx) / (dynPres * refArea)
        forces["cl"] = (forces.Fpy + forces.Fvy) / (dynPres * refArea)
        cpcf = pd.read_csv(os.path.join(os.path.join(fdir, "cyl.dat")),)

        avg_forces = forces[
            (args.tmin < forces.Time) & (forces.Time < args.tmax)
        ].mean()
        avg_forces["Re"] = rho * uInfty * cylDiameter / mu

        plt.figure("cd")
        plt.plot(
            forces.Time, forces.cd, lw=2, color=cmap[i], label=f"Nalu-{model}",
        )

        plt.figure("cl")
        plt.plot(
            forces.Time, forces.cl, lw=2, color=cmap[i], label=f"Nalu-{model}",
        )

        plt.figure("cd_re")
        plt.plot(
            avg_forces.Re,
            avg_forces.cd,
            lw=2,
            ls="",
            marker=markertype[i],
            mfc=cmap[i],
            mec=cmap[i],
            label=f"Nalu-{model}",
        )

        plt.figure("cp")
        plt.plot(
            cpcf.theta,
            cpcf.pressure / dynPres,
            lw=2,
            color=cmap[i],
            label=f"Nalu-{model}",
        )

        plt.figure("cf")
        plt.plot(
            cpcf.theta, cpcf.tauw / dynPres, lw=2, color=cmap[i], label=f"Nalu-{model}",
        )

        # spectra
        for c in ["cl", "cd"]:
            idx = (args.tmin < forces.Time) & (forces.Time < args.tmax)
            f, y = getFFT(
                forces.Time[idx] * cylDiameter / uInfty, forces[idx][c], normalize=True,
            )
            plt.figure(f"fft_{c}")
            plt.loglog(
                f * cylDiameter / uInfty,
                abs(y),
                lw=2,
                color=cmap[i],
                label=f"Nalu-{model}",
            )
            print(
                f"{model}: predicted St based on {c} = {f[np.argmax(abs(y))]* cylDiameter / uInfty}."
            )

    pname = "plots.pdf"
    with PdfPages(pname) as pdf:

        plt.figure("cd")
        ax = plt.gca()
        plt.xlabel(r"$t u_\infty / D$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$C_d$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.ylim([0, 0.8])
        legend = ax.legend(loc="best")
        plt.tight_layout()
        pdf.savefig(dpi=300)

        plt.figure("cl")
        ax = plt.gca()
        plt.xlabel(r"$t u_\infty / D$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$C_l$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.ylim([-1.1, 1.1])
        legend = ax.legend(loc="best")
        plt.tight_layout()
        pdf.savefig(dpi=300)

        plt.figure("cd_re")
        ax = plt.gca()
        plt.xlabel(r"$Re$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$C_d$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        legend = ax.legend(loc="best")
        plt.tight_layout()
        pdf.savefig(dpi=300)

        plt.figure("cp")
        ax = plt.gca()
        plt.xlabel(r"$\theta$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$C_p$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.xlim([-1, 181])
        legend = ax.legend(loc="best")
        plt.tight_layout()
        pdf.savefig(dpi=300)

        plt.figure("cf")
        ax = plt.gca()
        plt.xlabel(r"$\theta$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$C_f$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.xlim([-1, 181])
        legend = ax.legend(loc="best")
        plt.tight_layout()
        pdf.savefig(dpi=300)

        plt.figure("fft_cd")
        ax = plt.gca()
        plt.axvline(0.37, linestyle="--", color="gray")
        plt.xlabel(r"$f D/u_\infty$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$A_d$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.xlim([1e-2, 2])
        plt.ylim([1e-8, 1e0])
        legend = ax.legend(loc="best")
        plt.tight_layout()
        pdf.savefig(dpi=300)

        plt.figure("fft_cl")
        ax = plt.gca()
        plt.axvline(0.37, linestyle="--", color="gray")
        plt.xlabel(r"$f D/u_\infty$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$A_l$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.xlim([1e-2, 2])
        plt.ylim([1e-8, 1e0])
        legend = ax.legend(loc="best")
        plt.tight_layout()
        pdf.savefig(dpi=300)
