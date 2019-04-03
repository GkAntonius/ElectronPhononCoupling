"""
Functions for adaptative q-point grid construction.
"""
from __future__ import print_function
import warnings
import numpy as np

__all__ = ['get_qpt_adaptative', 'get_qptgrid_adaptative']


def get_qpt_adaptative(symrel, gprim, qpt_c, wtq_c, ngqpt_c, ngqpt_f):
    """
    Given a coarse grid q-point grid, and the parameters of a fine q-point grid,
    generate a new grid that samples the q=0 small cell with the fine grid,
    and the rest of the BZ with the coarse grid.

    Arguments
    ---------

    symrel: np.array(nsym, 3, 3)
        List of symmetry matrices.
    gprim: np.array(3, 3)
        The reciprocal lattice vectors, where gprim[0,:] is the first vector.
    qpt_c: np.array(nqpt_c, 3)
        List of q-points on the coarse grid, in reduced coordinates.
    wtq_c: np.array(nqpt_c)
        Weight of the q-points on the coarse grid
    ngqpt_c: np.array(3)
        Number of divisions along each primitive vectors of reciprocate lattice
        for the coarse grid
    ngqpt_f: np.array(3)
        Number of divisions along each primitive vectors of reciprocate lattice
        for the fine grid

    Returns
    -------

    qpt_a: np.array(nqpt_a)
        List of q-points on the adaptative grid, in reduced coordinates.
    wtq_a: np.array(nqpt_a)
        Weight of the q-points on the adaptative grid
    qpt_minibz: np.array(nqpt_a)
        List of the q-points in qpt_a that fall in the mini BZ.
    wtq_minibz: np.array(nqpt_a)
        Weight of the q-points in qpt_a that fall in the mini BZ.
    """
    qpt_a = list()
    wtq_a = list()
    qpt_minibz = list()
    wtq_minibz = list()

    nqpt_c = len(wtq_c)

    N_f = np.prod(ngqpt_f)
    N_c = np.prod(ngqpt_c)

    wtq_c_s = np.array(wtq_c) / sum(wtq_c) * N_f

    ngqpt_c = np.array(ngqpt_c)
    ngqpt_f = np.array(ngqpt_f)

    fc_ratio = ngqpt_f / ngqpt_c

    if not np.allclose(ngqpt_f % ngqpt_c, 0):
        raise Exception('Fine grid must be a multiple of coarse grid in all directions.')

    if not np.allclose(fc_ratio % 2, 0):
        raise Exception('Ratio of fine-to-coarse grid must be a multiple of 2 in all directions.')

    # List of G vectors used to define the BZ boundary
    G_BZ_bound = get_G_BZ_bound(gprim)

    # List of G vectors used to define the mini BZ boundary
    G_mini_BZ_bound = list()
    for G in G_BZ_bound:
        G_mini_BZ_bound.append(G / ngqpt_c)

    # Metric for dot product in reduced coordinates
    gmet = np.dot(gprim, gprim.transpose())

    # (G ** 2) / 2 for the mini BZ boundary
    G2_2_mini_BZ_bound = list()
    for G in G_mini_BZ_bound:
        G2_2_mini_BZ_bound.append(0.5 * np.dot(G, np.dot(gmet, G)))

    # Precompute G * gmet for faster dot product
    G_mini_BZ_bound_gmet = list()
    for G in G_mini_BZ_bound:
        G_mini_BZ_bound_gmet.append(np.dot(gmet, G))

    def outside_minibz(q):
        """Is this q-point outside of the mini BZ?"""
        for G, G2_2 in zip(G_mini_BZ_bound_gmet, G2_2_mini_BZ_bound):
            qG = abs(np.dot(q, G))
            if qG - G2_2 > 1e-8:
                return True
        return False

    def nlim_minibz(q):
        """Count the number of boundaries of the mini BZ this qpoint lies on"""
        count = 0
        for G, G2_2 in zip(G_mini_BZ_bound_gmet, G2_2_mini_BZ_bound):
            qG = abs(np.dot(q, G))
            if np.isclose(qG, G2_2):
                count += 1
        return count

    # Add the coarse q-points, excluding Gamma
    for qpt, wtq in zip(qpt_c, wtq_c_s):

        if outside_minibz(qpt):
            qpt_a.append(np.array(qpt))
            wtq_a.append(wtq)

    # A star is a list of symmetry-equivalent qpoints. This is the list of stars.
    stars = list()

    # Iterate over fine q-points
    for i1 in range(fc_ratio[0]-1, -fc_ratio[0], -1):
        for i2 in range(fc_ratio[1]-1, -fc_ratio[1], -1):
            for i3 in range(fc_ratio[2]-1, -fc_ratio[2], -1):

                q = np.array([i1,i2,i3], dtype=np.float64) / ngqpt_f

                # Exclude points outside the mini BZ
                if outside_minibz(q):
                    continue

                # Exclude points already counted
                found = False
                for star in stars:
                    for qs in star:
                        if np.allclose(q, qs):
                            found = True
                            break
                    if found:
                        break
                if found:
                    continue

                # Compute the weights by applying all symmetries.
                # They are different from the weights obtained when using the fine grid
                # everywhere in the BZ, because these also account for larger q-points
                # that were flipped by TRS then translated to the mini BZ.
                star = list()
                w = 0.0
                for tr in (1, -1):
                    for S in symrel:

                        qp = tr * np.dot(S, q)
                        for qs in star:
                            if np.allclose(qp, qs):
                                break
                        else:
                            star.append(qp)
                            w += 1.0
                stars.append(star)

                # Scale the weights of the q-points that end up on the boundary
                # of the mini BZ
                nlim = nlim_minibz(q)
                w /= 2 ** nlim
                    
                # Add the q-point and its weight,
                # and make sure Gamma is the first element of the list
                if all(np.isclose(q, 0.)):
                    qpt_a.insert(0, q)
                    wtq_a.insert(0, w)
                    qpt_minibz.insert(0, q)
                    wtq_minibz.insert(0, w)
                else:
                    qpt_a.append(q)
                    wtq_a.append(w)
                    qpt_minibz.append(q)
                    wtq_minibz.append(w)

    # Perform some checks on the weights
    expected_weight_minibz = float(N_f) / N_c
    actual_weight_minibz = sum(wtq_minibz)
    if not np.isclose(expected_weight_minibz, actual_weight_minibz):
        warnings.warn('Expected mini BZ weight: {:.3f}\nActual: {:.3f}'.format(
                      expected_weight_minibz, actual_weight_minibz))
        #for w, q in zip(wtq_minibz, qpt_minibz):
        #    print(w, q)

    expected_weight_total = float(N_f)
    actual_weight_total = sum(wtq_a)
    if not np.isclose(expected_weight_total, actual_weight_total):
        warnings.warn('Expected total weight: {:.3f}\nActual: {:.3f}'.format(
                      expected_weight_total, actual_weight_total))

    wtq_a = np.array(wtq_a) / N_f

    qpt_a = np.array(qpt_a).tolist()
    wtq_a = np.array(wtq_a).tolist()
    qpt_minibz = np.array(qpt_minibz).tolist()
    wtq_minibz = np.array(wtq_minibz).tolist()

    return qpt_a, wtq_a, qpt_minibz, wtq_minibz


def get_qptgrid_adaptative(symmetries, qptgrid_coarse, ngqpt_f):
    """
    Get an adaptative qptgrid.

    Arguments
    ---------

    symmetries: dict
        symrel
        nsym
        tnons
    qptgrid_coarse: dict
        kpt
        wtk
        ngkpt
        gprim
    ngqpt_f: np.array(3)
        Number of divisions along each primitive vectors of reciprocate lattice
        for the fine grid

    Returns
    -------

    qptgrid_adaptative: dict
        ngkpt
        kpt
        wtk
        nkpt
        kpt_minibz
        wtk_minibz
        nkpt_minibz
        ngkpt_fine
        ngkpt_coarse
    """
    symrel = symmetries['symrel']
    
    gprim = qptgrid_coarse['gprim']
    qpt_c = qptgrid_coarse['kpt']
    wtq_c = qptgrid_coarse['wtk']
    ngqpt_c = qptgrid_coarse['ngkpt']
    
    qpt_a, wtq_a, qpt_minibz, wtq_minibz = get_qpt_adaptative(symrel, gprim, qpt_c, wtq_c, ngqpt_c, ngqpt_f)
    
    qptgrid_adaptative = dict(
        ngkpt = ngqpt_f,
        kpt = qpt_a,
        wtk = wtq_a,
        nkpt = len(wtq_a),
        kpt_minibz = qpt_minibz,
        wtk_minibz = wtq_minibz,
        nkpt_minibz = len(wtq_minibz),
        ngkpt_fine = ngqpt_f,
        ngkpt_coarse = ngqpt_c,
        nshiftk = 1,
        shiftk = 3*[0.],
        )

    return qptgrid_adaptative


def get_G_BZ_bound(gprim):
    """
    Given the primitive vectors, find a set of G vectors
    that can be used to define the Brillouin Zone.
    """

    vol = np.linalg.det(gprim)
    if np.isclose(vol, 0.):
        raise Exception('Primitive vectors are not linearly independent.')

    gprim_scale = np.array(gprim) / vol

    gmet = np.dot(gprim_scale, gprim_scale.transpose())

    G_BZ_bound = [
        np.array([1, 0, 0], dtype=np.float),
        np.array([0, 1, 0], dtype=np.float),
        np.array([0, 0, 1], dtype=np.float),
        ]

    def outside_BZ(q, G_BZ_bound):
        for G in G_BZ_bound:
            Ggmet = np.dot(gmet, G)
            G2_2 = 0.5 * np.dot(G, Ggmet)
            qG = abs(np.dot(q, Ggmet))
            # This criterion is somewhat arbitrary,
            # but we want to make sure that the point is well inside
            # and not on the boundary
            if qG - G2_2 > -1e-3:
                return True
        return False

    G_BZ_bound_extend = np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
        [0, 1,-1],
        [1, 0,-1],
        [1,-1, 0],
        ], dtype=np.float)

    for Gnew in G_BZ_bound_extend:
        if not outside_BZ(Gnew/2, G_BZ_bound):
            G_BZ_bound.append(Gnew)

    return G_BZ_bound


