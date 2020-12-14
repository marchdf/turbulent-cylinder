# ========================================================================
#
# Imports
#
# ========================================================================
import argparse
import os
import numpy as np
import scipy.spatial.qhull as qhull
import pandas as pd
from mpi4py import MPI
import stk


# ========================================================================
#
# Functions
#
# ========================================================================
def p0_printer(par):
    iproc = par.rank

    def printer(*args, **kwargs):
        if iproc == 0:
            print(*args, **kwargs)

    return printer


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="A simple post-processing tool")
    parser.add_argument(
        "-m",
        "--mfile",
        help="Root name of files to postprocess",
        required=True,
        type=str,
    )
    parser.add_argument("--auto_decomp", help="Auto-decomposition", action="store_true")
    parser.add_argument(
        "--navg", help="Number of times to average", default=1, type=int
    )
    args = parser.parse_args()

    fdir = os.path.dirname(args.mfile)

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    par = stk.Parallel.initialize()
    printer = p0_printer(par)

    mesh = stk.StkMesh(par)
    printer("Reading meta data for mesh: ", args.mfile)
    mesh.read_mesh_meta_data(args.mfile, auto_decomp=args.auto_decomp)
    printer("Done reading meta data")

    printer("Loading bulk data for mesh: ", args.mfile)
    mesh.populate_bulk_data()
    printer("Done reading bulk data")

    num_time_steps = mesh.stkio.num_time_steps
    max_time = mesh.stkio.max_time
    tsteps = np.array(mesh.stkio.time_steps)
    printer(f"""Num. time steps = {num_time_steps}\nMax. time step  = {max_time}""")

    # Figure out the times over which to average
    tavg = tsteps[-args.navg :]
    printer("Averaging the following steps:")
    printer(tavg)

    # Extract time and spanwise average tau_wall on cylinder
    cyl_data = None
    for tstep in tavg:
        ftime, missing = mesh.stkio.read_defined_input_fields(tstep)
        printer(f"Loading tau_wall fields for time: {ftime}")

        coords = mesh.meta.coordinate_field
        wall = mesh.meta.get_part("cylinder")
        sel = wall & mesh.meta.locally_owned_part
        tauw = mesh.meta.get_field("tau_wall")
        pressure = mesh.meta.get_field("pressure")
        names = ["x", "y", "z", "tauw", "pressure"]
        nnodes = sum(bkt.size for bkt in mesh.iter_buckets(sel, stk.StkRank.NODE_RANK))

        cnt = 0
        data = np.zeros((nnodes, len(names)))
        for bkt in mesh.iter_buckets(sel, stk.StkRank.NODE_RANK):
            xyz = coords.bkt_view(bkt)
            tw = tauw.bkt_view(bkt)
            p = pressure.bkt_view(bkt)
            data[cnt : cnt + bkt.size, :] = np.hstack(
                (xyz, tw.reshape(-1, 1), p.reshape(-1, 1))
            )
            cnt += bkt.size

        if cyl_data is None:
            cyl_data = np.zeros(data.shape)
        cyl_data += data / len(tavg)

    lst = comm.gather(cyl_data, root=0)
    comm.Barrier()
    if rank == 0:
        df = pd.DataFrame(np.vstack(lst), columns=names)
        cyl = df.groupby("x", as_index=False).mean().sort_values(by=["x"])
        cyl["r"] = np.sqrt(cyl.x ** 2 + cyl.y ** 2)
        cyl["theta"] = (np.arctan2(cyl.x, cyl.y) + np.pi * 0.5) * 180 / np.pi
        cylname = os.path.join(fdir, "cyl.dat")
        cyl.to_csv(cylname, index=False)
