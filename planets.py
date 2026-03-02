from photochem.utils import stars

class Star:
    radius : float # relative to the sun
    Teff : float # K
    metal : float # log10(M/H)
    kmag : float
    logg : float
    planets : dict # dictionary of planet objects

    def __init__(self, radius, Teff, metal, kmag, logg, planets):
        self.radius = radius
        self.Teff = Teff
        self.metal = metal
        self.kmag = kmag
        self.logg = logg
        self.planets = planets
        
class Planet:
    radius : float # in Earth radii
    mass : float # in Earth masses
    Teq : float # Equilibrium T in K
    transit_duration : float # in seconds
    eclipse_duration: float # in seconds
    a: float # semi-major axis in AU
    stellar_flux: float # W/m^2
    
    def __init__(self, radius, mass, Teq, transit_duration, eclipse_duration, a, stellar_flux):
        self.radius = radius
        self.mass = mass
        self.Teq = Teq
        self.transit_duration = transit_duration
        self.eclipse_duration = eclipse_duration
        self.a = a
        self.stellar_flux = stellar_flux

# Pass et al. (2023), unless otherwise noted.

LTT1445Ab = Planet(
    radius=1.34,
    mass=2.73,
    Teq=431.0,
    transit_duration=1.366*60*60,
    eclipse_duration=1.366*60*60, # Assumed same as transit.
    a=0.03810,
    stellar_flux=stars.equilibrium_temperature_inverse(431.0, 0.0)
)

LTT1445A = Star(
    radius=0.271,
    Teff=3340, # Winters et al. (2022)
    metal=-0.34, # Winters et al. (2022)
    kmag=6.5, # Exo.Mast
    logg=4.97, # Exo.Mast
    planets={'b': LTT1445Ab}
)
