from src.setup_transmission_model import BeARTransmissionModel
import numpy as np
from astropy import units as u
from astropy import constants as const
import matplotlib.pyplot as plt
import pickle


# opacity paths for a bunch of species, update whenever fails to include missing ones.
# NOTE: these are relative to cross_section_file_path, BeAR prepends that itself.
opacity_paths = {
  'H2O': 'Molecules/1H2-16O_POKAZATEL',
  'CH4': 'Molecules/12C-1H4_HITEMP2020',
  'CO2': 'Molecules/12C-16O2_UCL-4000',
  'CO': 'Molecules/12C-16O_Li2015',
  'NH3': 'Molecules/14N-1H3_CoYuTe',
  'HCN': 'Molecules/1H-12C-14N_Harris',
  'H2S': 'Molecules/1H2-32S_AYT2',
  'SO2': 'Molecules/32S-16O2_ExoAmes',
  'CS2': 'Molecules/CS2_HITRAN',
  'H2CS': 'Molecules/1H2-12C-32S_MOTY',
  'H2CO': 'Molecules/1H2-12C-16O_AYTY',
  'C2H6': 'Molecules/12C2-1H6_HITRAN2020',
  'SiH4': 'Molecules/28Si-1H4_OY2T',
  'SiO': 'Molecules/28Si-16O_SiOUVenIR',
}

# species BeAR knows in chem_species.h that VULCAN networks commonly produce.
# these are used for the mean molecular weight / hydrostatic structure only,
# independently of whether we have an opacity for them. BeAR computes
# mu = sum(x_i * m_i) without renormalising, so leaving out a major species
# (e.g. atomic H in a dissociated upper atmosphere) makes mu -> 0, the scale
# height blow up, and the whole spectrum come back as NaN.
bear_chem_species = [
  'H', 'H2', 'He', 'O', 'C', 'OH', 'H2O', 'CO', 'CO2', 'CH3', 'CH4',
  'C2H2', 'C2H4', 'C2H6', 'H2CO', 'CH3OH', 'O2', 'HCN', 'N2', 'NH3',
  'NO2', 'N2O', 'CH3NH2', 'CH3CHO', 'CH3CN', 'SH', 'H2S', 'C6H6',
  'CS', 'CS2', 'NS', 'SO2', 'CH3SH', 'H2CS', 'CH3CCH',
]

def get_vulcan_data(vulcan_path):
  with open(vulcan_path, 'rb') as handle:
    data = pickle.load(handle)
    vulcan_spec = data['variable']['species']
    vulcan_pres = data['atm']['pco']/1.e6 # in bar
    vulcan_temp = data['atm']['Tco']
    vulcan_mixr = data['variable']['ymix']
    
    vulcan_data = {
      'spec': vulcan_spec,
      'pres': vulcan_pres,
      'temp': vulcan_temp,
      'mixr': vulcan_mixr}
  return vulcan_data

def truncate_atmosphere(vulcan_data, pressure_min, pressure_max):
  """Cut the VULCAN grid down to the pressure range BeAR can integrate.

  VULCAN routinely runs out to 1e-8 bar. With variable gravity the hydrostatic
  integration in BeAR diverges long before that (local g drops as (Rp/r)^2, so
  each step gets larger than the last), which gives infinite altitudes and a
  NaN spectrum. Those layers are optically thin anyway."""
  pressure = vulcan_data['pres']
  keep = (pressure <= pressure_max) & (pressure >= pressure_min)

  if not np.any(keep):
    raise ValueError("No VULCAN levels left after pressure truncation!")

  print("Using {} of {} VULCAN levels: {:.3e} - {:.3e} bar".format(
    np.sum(keep), pressure.size, pressure[keep].max(), pressure[keep].min()))

  return {
    'spec': vulcan_data['spec'],
    'pres': pressure[keep],
    'temp': vulcan_data['temp'][keep],
    'mixr': vulcan_data['mixr'][keep, :]}

def get_vulcan_params(vulcan_toml):
  import tomllib

  with open(vulcan_toml, "rb") as f:
      vulcan_params = tomllib.load(f)

  return vulcan_params
  
def get_species_data(vul_species, 
                     species_to_plot=[
                       'H2O',
                       'CH4',
                       'CO2',
                       'CO',
                       'NH3',
                       'HCN',
                       'H2S',
                       'SO2',
                       'CS2'],
                       include_rayleigh=True):
  # start out with H2He as default contributions
  species_data = [['CIA-H2-H2', 'CIA/H2-H2'], 
                  ['CIA-H2-He', 'CIA/H2-He']]
  # add all species with available opacities that are in species_to_plot
  for s in vul_species:
    if s in species_to_plot:
      try:
        species_data.append([s, opacity_paths[s]])
      except KeyError:
        print("Could not find opacity path for species", s)
        print("Continuing... but you should add the path and rerun if there is one!")
        continue

  if 'H2O' in vul_species and include_rayleigh:
    species_data.append(['H2O', 'Rayleigh'])
  if 'CO' in vul_species and include_rayleigh:
      species_data.append(['CO', 'Rayleigh'])
  if 'CO2' in vul_species and include_rayleigh:
      species_data.append(['CO2', 'Rayleigh'])
  if 'CH4' in vul_species and include_rayleigh:
      species_data.append(['CH4', 'Rayleigh'])

  return np.array(species_data)

def VMR_check(mixing_ratios):
  total_vmr = np.sum(mixing_ratios, axis=1)
  print('Average total VMR: ', np.mean(total_vmr), ' +/- ', np.std(total_vmr))
  if any(total_vmr < 0.9):
    print('Warning: missing some major species through summing VMRs!')
    print('          ', np.sum(total_vmr < 0.9), ' points have less than 90% total VMR.')
  elif any(total_vmr < 0.99):
    print('Warning: missing some minor species through summing VMRs!')
    print('          ', np.sum(total_vmr < 0.99), ' points have less than 99% total VMR.')
  if any(total_vmr > 1.01):
    print('Warning: too many molecules through summing VMRs!')
    print('          ', np.sum(total_vmr > 1.01), ' points have more than 101% total VMR.')
  return

def set_clouds(cloud_optical_depth, cloud_pressure, pressure, grid_points_number, transmission_model):
  cloud_tau = np.zeros((grid_points_number-1, transmission_model.wavelengths.size))

  if cloud_pressure is not None and cloud_optical_depth > 0.0:
    # find the layer where the cloud top sits. pressure runs from the bottom of
    # the atmosphere upwards, so it is descending and searchsorted needs the
    # reversed array.
    cloud_start = pressure.size - np.searchsorted(pressure[::-1], cloud_pressure)
    cloud_start = min(cloud_start, grid_points_number-1)

    # everything below the cloud top is opaque
    cloud_tau[:cloud_start, :] = cloud_optical_depth
  else:
    print("No clouds due to your settings.")

  return cloud_tau

def run_vulcan_model(vulcan_path,
                      vulcan_toml,
                      species_to_plot=[
                       'H2O',
                       'CH4',
                       'CO2',
                       'CO',
                       'NH3',
                       'HCN',
                       'H2S',
                       'SO2',
                       'CS2'],
                       include_rayleigh=True,
                       cloud_optical_depth=0.0,
                       cloud_pressure=None,
                       pressure_min=1e-5,
                       pressure_max=1e2,
                       wl_min=0.4,
                       wl_max=10.0,
                       resolution=10000.0,
                       cross_section_file_path="/work2/lbuc/lukas/opacities/",
                       wavenumber_path='/work2/lbuc/lukas/opacities/wavenumber_full.dat',
                       use_gpu=True):

  # set up the transmission model
  spectral_discretisation = 'const_resolution'

  vulcan_data = get_vulcan_data(vulcan_path)
  vulcan_data = truncate_atmosphere(vulcan_data, pressure_min, pressure_max)

  opacity_species_data = get_species_data(vulcan_data['spec'], 
                                          species_to_plot, 
                                          include_rayleigh)

  grid_points_number = len(vulcan_data['pres'])
  
  transmission_model = BeARTransmissionModel(
    use_gpu,
    grid_points_number,
    spectral_discretisation,
    wl_min,
    wl_max,
    resolution,
    cross_section_file_path, 
    opacity_species_data,
    wavenumber_path)

  # set up the atmosphere
  vulcan_params = get_vulcan_params(vulcan_toml)
  atm_params = vulcan_params['atmosphere']
  surface_gravity = atm_params['gs']   # cgs units
  planet_radius = atm_params['Rp']     # cgs units
  radius_ratio = planet_radius / (atm_params['r_star'] * const.R_sun.cgs.value)
  pressure = vulcan_data['pres']
  temperature = vulcan_data['temp']

  # set up the chemistry. this list drives the mean molecular weight and hence
  # the hydrostatic structure, so it must contain every major species, not just
  # the ones we have opacities for.
  chem_species = list(bear_chem_species)
  # keep anything we explicitly want on top of the defaults
  for s in species_to_plot:
    if s not in chem_species:
      print("Warning: Species", s, "is not in BeAR's known species list, removed it!")
  for s in chem_species:
    if s not in vulcan_data['spec']:
      print("Info: Species", s, "not in vulcan data, removed it.")
  # build the kept list separately, removing from a list while iterating over it skips entries
  chem_species = [s for s in chem_species if s in vulcan_data['spec']]

  # BeAR wants one row of mixing ratios per grid point, not per layer
  mixing_ratios = np.zeros((grid_points_number, len(chem_species)))
  for i, s in enumerate(chem_species):
    mixing_ratios[:,i] = vulcan_data['mixr'][:,vulcan_data['spec'].index(s)]

  # check if we're missing some major species through summing VMRs
  VMR_check(mixing_ratios)

  cloud_optical_depth = set_clouds(cloud_optical_depth, cloud_pressure, pressure, grid_points_number, transmission_model)

  # run the model
  spectrum = transmission_model.calcSpectrum(
    surface_gravity, 
    planet_radius, 
    radius_ratio, 
    pressure, 
    temperature, 
    chem_species, 
    mixing_ratios,
    cloud_optical_depth,
    use_variable_gravity=True)

  if np.any(~np.isfinite(spectrum)):
    print("Warning: spectrum contains non-finite values!")
    print("         this usually means the hydrostatic structure diverged:")
    print("         check the total VMR above and/or lower pressure_min.")

  return transmission_model, spectrum

def plot(transmission_model,
         spectrum,
         name,
         ylim=None):
  
  fig, ax = plt.subplots()
  ax.plot(transmission_model.wavelengths, spectrum)

  print(transmission_model.wavelengths)
  print(spectrum)

  ax.set_ylim(ylim)
  ax.set_xscale('log')

  plt.xlabel("Wavelength ($\mu$m)")
  plt.ylabel("Transit depth (ppm)")
  plt.savefig(f"/work2/lbuc/lukas/Projects/neoVULCAN/neoVULCAN/plot/{name}.png", dpi=300)
  plt.clf()

  return

if __name__ == "__main__":
  ##############################################################################################
  #                             edit these to change to new run                                #
  ##############################################################################################

  # name = 'HR858b_1x_8Kzz'
  # vulcan_path = "/work2/lbuc/lukas/Projects/neoVULCAN/neoVULCAN/output/{}.vul".format(name)
  # vulcan_toml = "/work2/lbuc/lukas/Projects/neoVULCAN/neoVULCAN/vulcan_{}.toml".format(name)
  name = 'HR858d_1x_8Kzz'
  # name = 'HR858d_1x_6Kzz'
  # name = 'HR858d_1x_10Kzz'
  vulcan_path = "/work2/lbuc/lukas/Projects/neoVULCAN/neoVULCAN/output/{}.vul".format(name)
  vulcan_toml = "/work2/lbuc/lukas/Projects/neoVULCAN/neoVULCAN/vulcan_{}.toml".format(name)

  transmission_model, spectrum = run_vulcan_model(vulcan_path, 
                                                  vulcan_toml,
                                                  resolution=100)

  plot(transmission_model, 
       spectrum, 
       name+'_spectrum',
       ylim=[230, 400])
