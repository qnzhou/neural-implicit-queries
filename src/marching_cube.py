#!/usr/bin/env python

import argparse
import mlp
import implicit_mlp_utils
import numpy as np
import kd_tree
import lagrange
import os

if not os.environ.get("JAX_ENABLE_X64", False):
    raise RuntimeError("JAX_ENABLE_X64 must be set to 1 for this script to work")


def parse_args():
    parser = argparse.ArgumentParser(description="Marching cube algorithm")
    parser.add_argument("input", help="Input neural implicit weights")
    parser.add_argument("output", help="Output mesh file")
    parser.add_argument("-d", "--depth", type=int, default=6, help="Grid depth")
    return parser.parse_args()


def main():
    args = parse_args()

    implicit_func, params = implicit_mlp_utils.generate_implicit_from_file(
        args.input, mode="sdf"
    )

    vertices = kd_tree.hierarchical_marching_cubes(
        implicit_func, params, np.ones(3) * -1, np.ones(3), args.depth
    )

    mesh = lagrange.SurfaceMesh()
    mesh.vertices = vertices.reshape((-1, 3)).astype(np.float64)
    mesh.facets = np.arange(len(vertices) * 3).reshape((-1, 3))
    lagrange.io.save_mesh(args.output, mesh)


if __name__ == "__main__":
    main()
