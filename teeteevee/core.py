import numpy as np
import astropy.units as u
from scipy.special import ellipk, ellipkinc
from scipy.integrate import quad

__all__ = ['Planet', 'delta_t_1']


class Planet(object):
    def __init__(self, mass, period, a, lam):
        """
        Define properties of a planet

        Parameters
        ----------
        mass : float
        period : float
        a : float
        lam : float
        """
        self.mass = mass
        self.period = period
        self.a = a
        self.lam = lam


def F(theta, k):
    """
    Elliptic integral of the first kind.

    Parameters
    ----------
    theta : float
    k : float
    """
    return ellipkinc(theta, k)


def K(k):
    """
    Complete elliptical integral of the first kind.

    Parameters
    ----------
    k : float

    Returns
    -------
    """
    return ellipk(k)


def Q(psi, alpha, N=2):
    theta_tilda = (psi + np.pi) / 2 - np.pi / 2 * N
    k = 2 * np.sqrt(alpha) / (1 + alpha)

    if N % 2 == 0:
        n_term = (N - 1) * K(k) + F(theta_tilda, k)
    else:
        n_term = N * K(k) - F(np.pi/2 - theta_tilda, k)

    return 2 / (1 + alpha) * n_term


def b_half(j, alpha):
    integrand = lambda theta: (np.cos(j * theta) /
                               (np.pi * (1 - 2 * alpha * np.cos(theta) +
                                         alpha**2)**0.5))
    return quad(integrand, 0, 2*np.pi)[0]


def db_half_dalpha(j, alpha):
    return -1.0 * quad(lambda theta: (alpha - np.cos(theta)) *
                                      np.cos(j * theta) / (alpha ** 2 -
                                                           2 * alpha *
                                                           np.cos(theta) +
                                                           1) ** 1.5,
                       0, 2*np.pi)[0] / np.pi


def c_j(plus_or_minus, j, alpha):
    return (alpha * db_half_dalpha(j, alpha) +
            plus_or_minus * 2 * j * b_half(j, alpha))


def delta_t_1(planet1, planet2, stellar_mass):
    """
    Compute the amplitude of TTVs due to planets on nearly circular orbits,
    following Equation A7 of Agol et al. (2005).

    Parameters
    ----------
    planet1 : `~teeteevee.Planet`
        Planet parameters

    planet2 : `~teeteevee.Planet`
        Planet parameters

    stellar_mass : `~astropy.units.Quantity`
        Mass of the host star

    Returns
    -------

    """
    alpha = float(planet1.a / planet2.a)
    n_1 = 2 * np.pi / planet1.period
    n_2 = 2 * np.pi / planet2.period
    psi = (planet1.lam - planet2.lam).to(u.rad).value
    k = 2 * np.sqrt(alpha) / (1 + alpha)

    first_term = (3 * planet2.mass / stellar_mass *
                  alpha * n_1 / (n_1 - n_2)**2) * (Q(psi, alpha) -
                                                   (2 * psi * K(k)) /
                  (np.pi * (1 + alpha)) - alpha * np.sin(psi))

    lam_10 = np.pi / 2
    second_term_prefactor = n_1 * planet2.mass / stellar_mass * alpha
    second_term_0 = (c_j(+1, 0, alpha) /
                     n_1**2 * np.sin(planet1.lam.to(u.rad).value - lam_10))

    return first_term.decompose()
