"""
Functions for adaptative q-point grid construction.

"""
from __future__ import print_function
import os
import pickle
import warnings
import numpy as np

__all__ = ['get_qpt_adaptative', 'get_qptgrid_adaptative']


def get_qpt_adaptative(symrel, gprim, qpt_c, wtq_c, ngqpt_c, ngqpt_f):
    """
    Given a coarse grid and a fine grid, generate a new grid that
    sample the q=0 small cell with the fine grid, and the rest
    of the BZ with the coarse grid.

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
    #nqpt_f = len(wtq_f)

    N_f = np.prod(ngqpt_f)
    N_c = np.prod(ngqpt_c)

    #wtq_f = np.array(wtq_f) / sum(wtq_f) * N_f
    wtq_c_s = np.array(wtq_c) / sum(wtq_c) * N_f

    ngqpt_c = np.array(ngqpt_c)
    ngqpt_f = np.array(ngqpt_f)

    fc_ratio = ngqpt_f / ngqpt_c

    if not np.allclose(ngqpt_f % ngqpt_c, 0):
        raise Exception('Fine grid must be a multiple of coarse grid in all directions.')

    if not np.allclose(fc_ratio % 2, 0):
        raise Exception('Ratio of fine-to-coarse grid must be a multiple of 2 in all directions.')

    gmet = np.dot(gprim, gprim.transpose())
    minibz_gmet = gmet / ngqpt_c

    #minibz_gprim_red = np.identity(3) / ngqpt_c

    # G_i ** 2 / 2, where G_i are the primitive vectors of the mini BZ
    minibz_g2 = np.array([0.5 * gmet[i,i] / ngqpt_c[i] ** 2 for i in range(3)])

    def outside_minibz(minibz_dots, minibz_g2):
        """Is this q-point outside of the mini BZ?"""
        return any(minibz_dots[i] - minibz_g2[i] > 1e-8 for i in range(3))

    def nlim_minibz(minibz_dots, minibz_g2):
        """Count the number of boundaries of the mini BZ this qpoint lies on"""
        return np.count_nonzero(np.isclose(minibz_dots, minibz_g2))

    for qpt, wtq in zip(qpt_c, wtq_c_s):

        minibz_dots = np.abs(np.dot(minibz_gmet, qpt))

        if outside_minibz(minibz_dots, minibz_g2):
            qpt_a.append(np.array(qpt))
            wtq_a.append(wtq)

    stars = list()
    for i1 in range(fc_ratio[0]):
        for i2 in range(fc_ratio[1]):
            for i3 in range(fc_ratio[2]):

                q = np.array([i1,i2,i3], dtype=np.float64) / ngqpt_f

                # Make sure that all components are as small as possible.
                #q = np.array(qpt)
                #for i in range(3):
                #    if abs(q[i]) > 0.5:
                #        q[i] -= np.sign(q[i]) * 1.0
                #if np.sign(q[0]) < 0 :
                #    q *= -1

                minibz_dots = np.abs(np.dot(minibz_gmet, q))

                # Exclude points outside the mini BZ
                if outside_minibz(minibz_dots, minibz_g2):
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
                # The original weights on the fine grid also account
                # for larger q-points that were flipped by TRS
                # then translated to the mini BZ, and we don't want that.
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
                nlim = nlim_minibz(minibz_dots, minibz_g2)
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

