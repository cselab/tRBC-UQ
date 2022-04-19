#!/usr/bin/env python

import numpy as np
import trimesh

def center_align_mesh(mesh: trimesh.Trimesh):
    mesh.vertices -= np.mean(mesh.vertices, axis=0)
    data = mesh.vertices

    eigenvectors, eigenvalues, V = np.linalg.svd(data.T, full_matrices=False)
    projected_data = np.dot(data, eigenvectors)
    mesh.vertices = projected_data
    return mesh


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_in', type=str, help='Input mesh')
    parser.add_argument('mesh_out', type=str, help='Output mesh')
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh_in, process=False)
    mesh = center_align_mesh(mesh)
    mesh.export(args.mesh_out)
