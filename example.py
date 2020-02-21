from teeteevee import Planet, delta_t_1 as ttv
import astropy.units as u
import numpy as np

lams = np.arange(0, 360)

p1 = Planet(4.45 * u.M_earth, 13.83989 * u.day, 0.1153 * u.AU, 0*u.deg)
p2 = Planet(8.08 * u.M_earth, 16.23855 * u.day, 0.1283 * u.AU, lams*u.deg)
stellar_mass = 1.071 * u.M_sun

ttvs = ttv(p1, p2, stellar_mass)

print(ttvs.max().to(u.hour))
