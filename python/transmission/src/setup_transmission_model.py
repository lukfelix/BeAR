import os
import sys

current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)
sys.path.append(os.path.dirname(parent_directory))

from lib import bear
import numpy as np


valid_spectral_discretisations = {'const_wavenumber', 'const_wavelength', 'const_resolution'}


class BeARTransmissionModel:

  def __init__(
      self,
      use_gpu,
      grid_points_number,
      spectral_discretisation,
      wavelength_min,
      wavelength_max,
      resolution,
      cross_section_file_path, 
      opacity_species_data,
      wavenumber_file_path = None) :
    
    if spectral_discretisation not in valid_spectral_discretisations:
      raise ValueError("Spectral discretisation must be one of %r." % valid_spectral_discretisations)
    
    self.nb_grid_points = grid_points_number
    
    bear_config = bear.Config()
    
    bear_config.use_gpu = np.bool_(use_gpu)
    bear_config.forward_model_type = "transmission"
    bear_config.cross_section_file_path = cross_section_file_path

    if wavenumber_file_path is not None:
      bear_config.wavenumber_file_path = wavenumber_file_path
    else :
      bear_config.wavenumber_file_path = ""
    
    if spectral_discretisation == 'const_wavenumber':
      bear_config.spectral_disecretisation = 0

    if spectral_discretisation == 'const_wavelength':
      bear_config.spectral_disecretisation = 1

    if spectral_discretisation == 'const_resolution':
      bear_config.spectral_disecretisation = 2

    bear_config.spectral_resolution = resolution
   
    self.spectral_grid = bear.SpectralGrid(
      bear_config,
      wavelength_min,
      wavelength_max)
    
    self.wavelengths = np.flip(np.array(self.spectral_grid.wavelength_list))
    self.wavenumbers = np.flip(np.array(self.spectral_grid.wavenumber_list))

    opacity_species = opacity_species_data[:, 0]
    opacity_folders = opacity_species_data[:, 1]

    self.forward_model = bear.TransmissionModel(
      bear_config, 
      self.spectral_grid, 
      self.nb_grid_points, 
      opacity_species, 
      opacity_folders)
  

  def calcSpectrum(
    self, 
    surface_gravity, 
    planet_radius, 
    radius_ratio, 
    pressure, 
    temperature, 
    chem_species, 
    mixing_ratios,
    cloud_optical_depth,
    use_variable_gravity=False) :

    cloud_tau = np.copy(cloud_optical_depth)
    
    #reverse the cloud optical depth array because BeAR uses wavenumbers in increasing order
    for i in range(cloud_tau.shape[0]):
      cloud_tau[i] = np.flip(cloud_tau[i])

    spectrum = np.array(
      self.forward_model.calcSpectrum(
        surface_gravity, 
        planet_radius, 
        radius_ratio, 
        pressure, 
        temperature, 
        chem_species, 
        mixing_ratios, 
        cloud_tau,
        np.bool_(use_variable_gravity)))
    
    spectrum = np.flip(spectrum)

    return spectrum
  
  # def convolve_spectrum(self, spectrum, FWHM_vals):
  #   """Convolve the input spectrum with a Gaussian kernel of given FWHM values. Cut off the convolution at 5 sigma."""
  #   convolved_spectrum = np.zeros_like(spectrum)
  #   for i in range(len(spectrum)):
  #       FWHM = FWHM_vals[i]
  #       sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
  #       if i<len(spectrum)-1:
  #         kernel_size = int(5 * sigma / (self.wavelengths[i+1] - self.wavelengths[i]))  # cut off at 5 sigma
  #         kernel_x = np.arange(-kernel_size, kernel_size + 1) * (self.wavelengths[i+1] - self.wavelengths[i])
  #       else: # need to approximate last point's FWHM using the previous point, otherwise kernel size is not defined at all, should be fine though
  #         kernel_size = int(5 * sigma / (self.wavelengths[i] - self.wavelengths[i-1]))  # cut off at 5 sigma
  #         kernel_x = np.arange(-kernel_size, kernel_size + 1) * (self.wavelengths[i] - self.wavelengths[i-1])
  #       kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
  #       kernel /= np.sum(kernel)  # normalize the kernel
  #       convolved_spectrum[i] = np.sum(spectrum[max(0, i - kernel_size):min(len(spectrum), i + kernel_size + 1)] * kernel[max(0, kernel_size - i):min(2 * kernel_size + 1, kernel_size + len(spectrum) - i)])
  #   return convolved_spectrum