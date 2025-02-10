import numpy as np
import scipy.sparse as sparse

from tqdm.auto import tqdm

import densemaps.numpy.maps as maps  # type: ignore

from ..geometry_utils import compute_embedding


def rhm_refine(
    mesh1,
    mesh2,
    P12_init,
    P21_init,
    alpha=5e-4,
    beta=5e-3,
    emb_dim=8,
    nit_max=200,
    nit_min=20,
    abs_tol=1e-9,
    n_jobs=1,
    log=False,
    precise=True,
    last_precise=True,
    verbose=False,
):
    r"""Refines a functional map using the RHM algorithm.

    This solves for the optimal pointwise map between two meshes using the RHM algorithm using the RHM energy:

    $E_{RHM} = \alpha E_{dirichlet} + (1-\alpha)E_{bij} + \beta E_{coupling}$$

    Parameters
    -----------------------------
    mesh1 : TriMesh
        The source mesh.
    mesh2 : TriMesh
        The target mesh.
    P12_init : PointWiseMap
        The initial map from mesh1 to mesh2. (N1, N2)
    P21_init : PointWiseMap
        The initial map from mesh2 to mesh1. (N2, N1)
    alpha : float
        The weight of the harmonic energy term. Bijectivity is weighted by 1-alpha.
    beta : float
        The weight of the penalty energy term.
    emb_dim : int
        The dimension of the embedding space for vertices (distance mimics the geodesic distance).
    nit_max : int
        The maximum number of iterations.
    nit_min : int
        The minimum number of iterations.
    abs_tol : float
        The absolute tolerance for the stopping criterion.
    n_jobs : int
        The number of parallel jobs to run for NN queries
    log : bool
        Whether to return the values of the energy at each step.
    precise : bool
        Whether to use the precise computation for the energy terms (else, use simple nn queries).
    last_precise : bool
        Whether to use the precise computation for the last iteration. Only used if precise is False. Else always True
    verbose : bool
        Whether to display a progress bar.

    Returns
    -----------------------------
    P12 : PointWiseMap
        The refined map from mesh1 to mesh2.
    P21 : PointWiseMap
        The refined map from mesh2 to mesh1.
    energy_log : list, optional
        The energy values at each iteration if log is True.

    """

    assert 0 < alpha < 1, "alpha must be in (0,1)"

    X1 = compute_embedding(mesh1, n_samples=500, n_components=emb_dim, n_jobs=n_jobs)
    X2 = compute_embedding(mesh2, n_samples=500, n_components=emb_dim, n_jobs=n_jobs)

    res = rhm_refine_fast(
        mesh1,
        mesh2,
        X1,
        X2,
        P12_init,
        P21_init,
        alpha=alpha,
        beta=beta,
        nit_max=nit_max,
        nit_min=nit_min,
        abs_tol=abs_tol,
        n_jobs=n_jobs,
        log=log,
        precise=precise,
        last_precise=last_precise,
        verbose=verbose,
    )

    return res


def rhm_refine_fast(
    mesh1,
    mesh2,
    X1,
    X2,
    P12_init,
    P21_init,
    alpha=5e-4,
    beta=5e-3,
    nit_max=200,
    nit_min=20,
    abs_tol=1e-9,
    n_jobs=1,
    log=False,
    precise=True,
    last_precise=True,
    verbose=False,
):
    """
    Solve RHM refinement with embeddings

    Parameters
    -----------------------------
    mesh1 : TriMesh
        The source mesh.
    mesh2 : TriMesh
        The target mesh.
    X1 : np.ndarray
        The embedding of mesh1 vertices. (N1, p)
    X2 : np.ndarray
        The embedding of mesh2 vertices. (N1, p)
    P12_init : PointWiseMap
        The initial map from mesh1 to mesh2. (N1, N2)
    P21_init : PointWiseMap
        The initial map from mesh2 to mesh1. (N2, N1)
    alpha : float
        The weight of the harmonic energy term. Bijectivity is weighted by 1-alpha.
    beta : float
        The weight of the penalty energy term.
    emb_dim : int
        The dimension of the embedding space for vertices (distance mimics the geodesic distance).
    nit_max : int
        The maximum number of iterations.
    nit_min : int
        The minimum number of iterations.
    abs_tol : float
        The absolute tolerance for the stopping criterion.
    n_jobs : int
        The number of parallel jobs to run for NN queries
    log : bool
        Whether to return the values of the energy at each step.
    precise : bool
        Whether to use the precise computation for the energy terms (else, use simple nn queries).
    last_precise : bool
        Whether to use the precise computation for the last iteration. Only used if precise is False. Else always True
    verbose : bool
        Whether to display a progress bar.

    Returns
    -----------------------------
    P12 : PointWiseMap
        The refined map from mesh1 to mesh2.
    P21 : PointWiseMap
        The refined map from mesh2 to mesh1.
    energy_log : list, optional
        The energy values at each iteration if log is True.

    """

    P12 = P12_init
    P21 = P21_init

    X12 = P12 @ X2  # (n1,d)
    X21 = P21 @ X1  # (n2,d)

    energy_log = []

    # Beta list as defined in the Original RHM code
    beta_list = np.clip(beta * np.arange(1, nit_max + 1), None, 100 * beta)

    iterable = tqdm(range(nit_max)) if verbose else range(nit_max)
    for iterind in iterable:
        beta_cur = beta_list[iterind]

        P21 = solve_P12(
            X2,
            X1,
            X21,
            X12,
            mesh2.area,
            mesh1.area,
            alpha,
            beta_cur,
            faces2=mesh1.faces,
            n_jobs=n_jobs,
            precise=precise,
        )
        X21 = solve_X12_fast(
            X1,
            P21,
            P12,
            mesh2.vertex_areas,
            mesh1.vertex_areas,
            mesh2.W,
            alpha,
            beta_cur,
        )

        energy_log.append(
            get_energies(mesh1, mesh2, X1, X2, P12, P21, X12, X21, alpha, beta_cur)
        )  # 2*iterind

        P12 = solve_P12(
            X1,
            X2,
            X12,
            X21,
            mesh1.area,
            mesh2.area,
            alpha,
            beta_cur,
            faces2=mesh2.faces,
            n_jobs=n_jobs,
            precise=precise,
        )
        X12 = solve_X12_fast(
            X2,
            P12,
            P21,
            mesh1.vertex_areas,
            mesh2.vertex_areas,
            mesh1.W,
            alpha,
            beta_cur,
        )

        energy_log.append(
            get_energies(mesh1, mesh2, X1, X2, P12, P21, X12, X21, alpha, beta_cur)
        )  # 2*iterind

        if verbose:
            iterable.set_description(
                f"Iter : {iterind} - Harmonic : {energy_log[-1][0]:.3e}"
            )

        if iterind > nit_min:
            crit = np.abs(energy_log[-1][-1] - energy_log[-10][-1]) < abs_tol

            if crit:
                P21 = solve_P12(
                    X2,
                    X1,
                    X21,
                    X12,
                    mesh2.area,
                    mesh1.area,
                    alpha,
                    beta_cur,
                    faces2=mesh1.faces,
                    n_jobs=n_jobs,
                    precise=last_precise,
                )
                P12 = solve_P12(
                    X1,
                    X2,
                    X12,
                    X21,
                    mesh1.area,
                    mesh2.area,
                    alpha,
                    beta_cur,
                    faces2=mesh2.faces,
                    n_jobs=n_jobs,
                    precise=last_precise,
                )
                break

    if log:
        return P12, P21, energy_log

    return P12, P21


def solve_P12(
    X1, X2, X12, X21, area1, area2, alpha, beta, n_jobs=1, precise=True, faces2=None
):
    r"""Solve for the pointwise map

    Solves $(1-\alpha) E_{bij}(P_{12}, X_{21}, X_1) + \beta * E_{coupling}(P_{12}, X_2, X_{12})$

    Expresses the problem in the form $\min_{P_{12}}||P_{12} @ A - B||_2^2

    where $A = w_{bij} * X_2$ and $B = w_{bij} * X_1 + w_{couple} * X_{12}$

    Parameters
    -----------------------------
    X1 : np.ndarray
        The embedding of mesh1 vertices. (N1, p)
    X2 : np.ndarray
        The embedding of mesh2 vertices. (N2, p)
    X12 : np.ndarray
        The embedding of mesh1 vertices in the target space.  (N1, p)
    X21 : np.ndarray
        The embedding of mesh2 vertices in the source space.  (N2, p)
    area1 : float
        The total area of mesh1.
    area2 : float
        The total area of mesh2.
    alpha : float
        The weight of the harmonic energy term. Bijectivity is weighted by 1-alpha.
    beta : float
        The weight of the penalty energy term.
    n_jobs : int
        The number of parallel jobs to run for NN queries
    precise : bool
        Whether to use the precise computation for the energy terms (else, use simple nn queries).
    faces2 : np.ndarray, optional
        The faces of mesh2 if precise is True.

    Returns
    -----------------------------
    P12 : PointWiseMap
        The refined map from mesh1 to mesh2.  (N1, N2)

    """
    if precise:
        assert faces2 is not None, "Faces2 must be provided for precise computation"

    # Weight for each term
    w_bij = np.sqrt(1 - alpha) / area1  # cR
    w_couple = np.sqrt(beta) / np.sqrt(area1 * area2)  # cQ

    A_mat = np.concatenate([w_bij * X21, w_couple * X2], axis=1)  # (n2, 2d)
    B_mat = np.concatenate([w_bij * X1, w_couple * X12], axis=1)  # (n1, 2d)

    if precise:
        P12 = maps.EmbPreciseMap(A_mat, B_mat, faces2, n_jobs=n_jobs)  # (n1, n2)
    else:
        P12 = maps.EmbP2PMap(A_mat, B_mat, n_jobs=n_jobs)

    return P12


def solve_X12_fast(X2, P12, P21, vertex_areas1, vertex_areas2, W1, alpha, beta):
    r"""Solves for transferred embedding X12

    Solves $\alpha E_{dirichlet}(X_{12}) + (1-\alpha) E_{bij}(P_{21}, X_{12}, X_2) + \beta E_{coupling}(X_{12}, X_2, P_{12})$

    Expresses the problem in the form $\min_{X_{12}}||A X_{12} - B||_2^2

    And solves the dual problem $A^T A X_{12} = A^T B$

    Parameters
    -----------------------------
    X2 : np.ndarray
        The embedding of mesh2 vertices. (N2, p)
    P12 : PointWiseMap
        The refined map from mesh1 to mesh2.  (N1, N2)
    P21 : PointWiseMap
        The refined map from mesh2 to mesh1.  (N2, N1)
    vertex_areas1 : np.ndarray
        The area of each vertex in mesh1.  (N1,)
    vertex_areas2 : np.ndarray
        The area of each vertex in mesh2.  (N2,)
    W1 : sparse.csr_matrix
        The laplacian matrix of mesh1.  (N1, N1)
    alpha : float
        The weight of the harmonic energy term. Bijectivity is weighted by 1-alpha.
    beta : float
        The weight of the penalty energy term.

    Returns
    -----------------------------
    X12 : np.ndarray
        The refined embedding of mesh1 vertices in the target space.  (N1, p)

    """
    area1 = vertex_areas1.sum()
    area2 = vertex_areas2.sum()

    A1 = sparse.diags(vertex_areas1)
    A2 = sparse.diags(vertex_areas2)

    # Rescale weigthing for each term
    w_bij = (1 - alpha) / area2**2  # cR
    w_coup = beta / area1 * area2  # cQ
    w_dir = alpha / area2  # cD

    A_mat = w_bij * (P21.mT._to_sparse() @ A2 @ P21._to_sparse())
    A_mat += w_coup * A1
    A_mat += w_dir * W1

    B_mat = w_bij * (P21.mT @ (vertex_areas2[:, None] * X2))
    B_mat += w_coup * (A1 @ (P12 @ X2))

    X12 = sparse.linalg.spsolve(A_mat.tocsc(copy=False), B_mat)

    return X12


def harmonic_energy(P12X2, W1, area2):
    r"""Computes the harmonic energy

    $E_{dirichlet}(X_{12}) = \frac{1}{A_2} \|X_{12}\|_{W_1}$

    Parameters
    -----------------------------
    P12X2 : np.ndarray
        $P_{12} X_2$. The embedding of mesh1 vertices in the target space.  (N1, p)
    W1 : sparse.csr_matrix
        The laplacian matrix of mesh1.  (N1, N1)
    area2 : float
        The total area of mesh2.

    Returns
    -----------------------------
    energy : float
        The harmonic energy
    """

    energy = np.einsum("ij, ij-> i", P12X2, W1 @ P12X2).sum() / area2

    return energy


def penalty_energy(P12X2, X12, A1, area1, area2):
    r"""Computes the coupling energy

    $E_{coupling}(X_{12}, P_{12}X_2) = \frac{1}{A_1 A_2} \|X_{12} - P_{12}X_2\|_{A_1}^2$

    Parameters
    -----------------------------
    P12X2 : np.ndarray
        $P_{12} X_2$. The embedding of mesh1 vertices in the target space.  (N1, p)
    X12 : np.ndarray
        The embedding of mesh1 vertices in the target space.  (N1, p)
    A1 : sparse.csr_matrix
        The area matrix of mesh1.  (N1, N1)
    area1 : float
        The total area of mesh1.
    area2 : float
        The total area of mesh2.

    Returns
    -----------------------------
    energy : float
        The penalty energy

    """
    delta = X12 - P12X2

    energy = np.einsum("ij, ij-> i", delta, A1 @ delta).sum() / (area1 * area2)

    return energy


def bijectivity_energy(P12X2, P21, X2, A2, area2):
    r"""Computes the bijectivity energy

    $E_{bij}(P_{21}, X_{12}, X_2) = \frac{1}{A_2} \|X_{2} - P_{21}X_{12}\|_{A_2}^2$

    Parameters
    --------------------------------
    P12X2 : np.ndarray
        $P_{12} X_2$. The embedding of mesh1 vertices in the target space.  (N1, p)
    P21 : PointWiseMap
        The refined map from mesh2 to mesh1.  (N2, N1)
    X2 : np.ndarray
        The embedding of mesh2 vertices. (N2, p)
    A2 : sparse.csr_matrix
        The area matrix of mesh2.  (N2, N2)
    area2 : float
        The total area of mesh2.

    Returns
    --------------------------------
    energy : float
        The bijectivity energy

    """
    delta = X2 - P21 @ P12X2

    energy = np.einsum("ij, ij-> i", delta, A2 @ delta).sum() / (area2**2)

    return energy


def get_energies(mesh1, mesh2, X1, X2, P12, P21, X12, X21, alpha, beta):
    r"""Computes the energies

    Parameters
    -----------------------------
    mesh1 : TriMesh
        The source mesh.
    mesh2 : TriMesh
        The target mesh.
    X1 : np.ndarray
        The embedding of mesh1 vertices. (N1, p)
    X2 : np.ndarray
        The embedding of mesh2 vertices. (N2, p)
    P12 : PointWiseMap
        The refined map from mesh1 to mesh2.  (N1, N2)
    P21 : PointWiseMap
        The refined map from mesh2 to mesh1.  (N2, N1)
    X12 : np.ndarray
        The embedding of mesh1 vertices in the target space.  (N1, p)
    X21 : np.ndarray
        The embedding of mesh2 vertices in the source space.  (N2, p)
    alpha : float
        The weight of the harmonic energy term. Bijectivity is weighted by 1-alpha.
    beta : float
        The weight of the penalty energy term.

    Returns
    -----------------------------
    energies : list
        The harmonic, bijectivity, coupling and total energies

    """

    E_bij = (1 - alpha) * bijectivity_energy(X12, P21, X2, mesh2.A, mesh2.area)
    E_bij += (1 - alpha) * bijectivity_energy(X21, P12, X1, mesh1.A, mesh1.area)

    E_penalty = beta * penalty_energy(X12, P12 @ X2, mesh1.A, mesh1.area, mesh2.area)
    E_penalty += beta * penalty_energy(X21, P21 @ X1, mesh2.A, mesh2.area, mesh1.area)

    E_harmonic = alpha * harmonic_energy(X12, mesh1.W, mesh2.area)

    energies = [E_harmonic, E_bij, E_penalty, E_harmonic + E_bij + E_penalty]

    return energies
