import numpy as np

import scipy.sparse as sparse

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS

import pymeshlab
import igl

def knn_query(X, Y, k=1, return_distance=False, n_jobs=1):
    """
    Query nearest neighbors.

    Parameters
    ----------
    X : (n,d) np.ndarray
        Dataset
    Y : (m,d) np.ndarray
        Query points
    k : int
        Number of neighbors to return
    return_distance : bool
        Whether to return distances
    n_jobs : int
        Number of jobs to run in parallel

    Returns
    -------
    dists : (m,k) np.ndarray, optional
        Distances to the k nearest neighbors. Only returned if return_distance is True
    matches : (m,k) np.ndarray
        Indices of the k nearest neighbors
    """
    tree = NearestNeighbors(n_neighbors=k, leaf_size=40, algorithm="kd_tree", n_jobs=n_jobs)
    tree.fit(X)
    dists, matches = tree.kneighbors(Y)

    if k == 1:
        dists = dists.squeeze()
        matches = matches.squeeze()

    if return_distance:
        return dists, matches
    return matches

def decimate(mesh1, n_target_faces, n_jobs=1):
    """Runs quadratic mesh decimation on a mesh

    Parameters
    ----------
    mesh1 : TriMesh object
        input mesh
    n_target_faces : int
        number of faces in the output mesh
    n_jobs : int
        number of jobs to run in parallel for NN query

    Returns
    -------
    sub_indices : (m,) np.ndarray
        Indices of the vertices in the decimated mesh
    """


    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh1.vertlist, face_matrix=mesh1.facelist))
    ms.meshing_decimation_quadric_edge_collapse(preservetopology=True, targetfacenum=n_target_faces,
                                                optimalplacement=False)

    sub_vertices = ms.current_mesh().vertex_matrix()
    sub_indices = knn_query(mesh1.vertlist, sub_vertices, n_jobs=n_jobs)  # (m,)

    return sub_indices


def compute_embedding(mesh1, n_samples=500, n_components=8, n_jobs=1):
    """Compute embedding of a mesh using MDS

    The embedding mimics the geodesic distances of the mesh

    Parameters
    ----------
    mesh1 : TriMesh object
        input mesh
    n_samples : int
        number of samples to use for the embedding
    n_components : int
        number of components of the embedding
    n_jobs : int
        number of jobs to run in parallel for MDS

    Returns
    -------
    emb_final : (n,d) np.ndarray
        Embedding of the mesh

    """
    sub_indices = decimate(mesh1, 2 * n_samples)  # (m,)

    D_sub = mesh1.geod_from(sub_indices)[sub_indices, :]  # (m,m)
    Dmat = (D_sub+D_sub.T)/2

    # n_components = 8
    n_init = 4
    myMDS = MDS(n_components=n_components, n_init=n_init, dissimilarity='precomputed',
                max_iter=1000, n_jobs=min(n_jobs, n_init))
    emb1 = myMDS.fit_transform(Dmat)

    emb_init = np.zeros((mesh1.n_vertices, n_components))
    emb_init[sub_indices] = emb1

    A_mat = mesh1.W @ sparse.diags(1/mesh1.vertex_areas) @ mesh1.W
    B_mat = np.zeros((mesh1.n_vertices,n_components))

    res_min = igl.min_quad_with_fixed(A_mat, B_mat, sub_indices, emb1,
                                      sparse.csc_matrix(np.zeros((0, mesh1.n_vertices))),
                                      np.zeros((0, 1)),
                                      False)

    assert res_min[0], "Min quad failed"

    emb_final = res_min[1]

    return emb_final

