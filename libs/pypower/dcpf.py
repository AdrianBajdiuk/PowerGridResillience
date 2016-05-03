# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Solves a DC power flow.
"""

from numpy import copy, r_, matrix,linalg
from scipy.sparse.linalg import spsolve,lsqr
from scipy.linalg import det
import math

def dcpf(B, Pbus, Va0, ref, pv, pq):
    """Solves a DC power flow.

    Solves for the bus voltage angles at all but the reference bus, given the
    full system C{B} matrix and the vector of bus real power injections, the
    initial vector of bus voltage angles (in radians), and column vectors with
    the lists of bus indices for the swing bus, PV buses, and PQ buses,
    respectively. Returns a vector of bus voltage angles in radians.

    @see: L{rundcpf}, L{runpf}

    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln
    """
    pvpq = matrix(r_[pv, pq])

    ## initialize result vector
    Va = copy(Va0)

    ## update angles for non-reference buses
    A=B[pvpq.T, pvpq]
    d = det(A.todense())
    if(math.fabs(d) > 0.2):
        Va[pvpq] = spsolve(A, (Pbus[pvpq] - B[pvpq.T, ref] * Va0[ref]).transpose())
    else:
        Va[pvpq] = lsqr(A, (Pbus[pvpq] - B[pvpq.T, ref] * Va0[ref]).transpose())[0]

    return Va
