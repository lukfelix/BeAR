from src.setup_transmission_model import BeARTransmissionModel
import numpy as np
from astropy import units as u
from astropy import constants as const
import matplotlib.pyplot as plt
import dynesty
from dynesty import plotting as dyplot
from scipy.stats import norm, truncnorm
import pickle

cross_section_path = "/work2/lbuc/lukas/opacities/"
ndim = 14
lnorm = -0.5 * np.log(2 * np.pi) * ndim

class params:
    def __init__(self, surface_gravity, planet_radius, star_radius, 
                 temperature, p_bottom, p_top, mixing_ratios, 
                 rv_shift = 0.0):
        self.surface_gravity = surface_gravity
        self.planet_radius = planet_radius
        self.star_radius = star_radius
        self.temperature = temperature
        self.p_bottom = p_bottom
        self.p_top = p_top
        self.mixing_ratios = mixing_ratios
        self.rv_shift = rv_shift

def setup_retrieval_model(use_gpu=True, grid_points_number=60):
    #setting the basic properties of the model
    spectral_discretisation = 'const_resolution'
    wavelength_min = 0.4
    wavelength_max = 6.0
    resolution = 2000.0

    cross_section_file_path = cross_section_path
    wavenumber_path = '/work2/lbuc/lukas/opacities/wavenumber_full.dat'

    chem_species = np.array(['H2', 'He', 'H2O', 'CH4', 'CO2', 'CO', 'NH3', 'CS2', 'SO2'])
    opacity_species_data = np.array([
        ['CIA-H2-H2',   'CIA/H2-H2'], 
        ['CIA-H2-He',   'CIA/H2-He'],
        ['H2',          'Rayleigh'],
        ['He',          'Rayleigh'],
        ['H2O',     'Molecules/1H2-16O_POKAZATEL'],
        ['CH4',     'Molecules/12C-1H4_HITEMP2020'],
        ['CO2',     'Molecules/12C-16O2_UCL-4000'],
        ['CO',      'Molecules/12C-16O_Li2015'],
        ['NH3',     'Molecules/14N-1H3_CoYuTe'],
        ['CS2',     'Molecules/CS2_HITRAN'],
        ['SO2',     'Molecules/32S-16O2_ExoAmes']
        ])


    use_gpu = use_gpu
    # use_gpu = False
    grid_points_number = grid_points_number

    #create the BeAR forward model
    transmission_model = BeARTransmissionModel(
        use_gpu,
        grid_points_number,
        spectral_discretisation,
        wavelength_min,
        wavelength_max,
        resolution,
        cross_section_file_path, 
        opacity_species_data,
        wavenumber_path)
    return transmission_model, chem_species, grid_points_number

def get_FWHM_vals(instrument):
    if instrument in ['NIRISS', 'NIRISS_order1']:
        return np.loadtxt('/work2/lbuc/lukas/data/JWST/FWHM_data/FWHM_NIRISS_GR700XD_order1.dat')
    elif instrument in ['NIRISS2', 'NIRISS_order2']:
        return np.loadtxt('/work2/lbuc/lukas/data/JWST/FWHM_data/FWHM_NIRISS_GR700XD_order2.dat')
    elif instrument in ['NRS1', 'NRS2', 'G395H1', 'G395H2']:
        return np.loadtxt('/work2/lbuc/lukas/data/JWST/FWHM_data/FWHM_NIRSpec_G395H.dat')
    else:
        raise ValueError(f"Invalid instrument name: {instrument}\nProbably just not implemented yet, ask Lukas.")
    
def convolve_spectrum(wls, spectrum, FWHM_wls, FWHM_vals):
    """Convolve the input spectrum with a Gaussian kernel of given FWHM values. Cut off the convolution at 5 sigma."""
    convolved_spectrum = np.zeros_like(spectrum)
    for i in range(len(spectrum)):
        FWHM = np.interp(wls[i], FWHM_wls, FWHM_vals)
        sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
        if i<len(spectrum)-1:
          kernel_size = int(5 * sigma / (wls[i+1] - wls[i]))  # cut off at 5 sigma
          kernel_x = np.arange(-kernel_size, kernel_size + 1) * (wls[i+1] - wls[i])
        else: # need to approximate last point's FWHM using the previous point, otherwise kernel size is not defined at all, should be fine though
          kernel_size = int(5 * sigma / (wls[i] - wls[i-1]))  # cut off at 5 sigma
          kernel_x = np.arange(-kernel_size, kernel_size + 1) * (wls[i] - wls[i-1])
        kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
        kernel /= np.sum(kernel)  # normalize the kernel
        convolved_spectrum[i] = np.sum(spectrum[max(0, i - kernel_size):min(len(spectrum), i + kernel_size + 1)] * kernel[max(0, kernel_size - i):min(2 * kernel_size + 1, kernel_size + len(spectrum) - i)])
    return convolved_spectrum
    
def RV_shift_model(wls, shift):
    """Apply a radial velocity shift (in km/s) to the wavelength grid."""
    c = const.c.to('km/s').value
    delta_lambda = wls * (shift / c)
    return wls - delta_lambda

# def forward_model(params, model, grid_points_number, chem_species):
#     surface_gravity = params.surface_gravity
#     planet_radius = params.planet_radius * const.R_earth.cgs.value
#     radius_ratio = planet_radius / (params.star_radius * const.R_sun.cgs.value)
#     print(f"Radius ratio: {radius_ratio:.4f}")
#     print(f"planet radius: {planet_radius:.2e} cm")
#     print(f"star radius: {params.star_radius * const.R_sun.cgs.value:.2e} cm")
#     #p-T structure
#     temperature = np.full((grid_points_number), params.temperature)
#     pressure = np.logspace(params.p_bottom, params.p_top, grid_points_number)
#     #chemical composition
#     mix_ratios = np.zeros((grid_points_number, chem_species.size))  # also includes H2 and He, which will be set later
#     total_metals = np.sum(params.mixing_ratios)

#     # skip unphysical models
#     if total_metals >= 1.0:
#         return np.zeros(model.wavelengths.size)
    
#     H2He_ratio = 0.83/0.13
#     mix_ratios[:, 0] = (1.0 - total_metals) * H2He_ratio / (1.0 + H2He_ratio)
#     mix_ratios[:, 1] = (1.0 - total_metals) / (1.0 + H2He_ratio)
#     mix_ratios[:, 2:] = params.mixing_ratios

    
#     #cloud properties, set to zero for now
#     cloud_optical_depth = np.zeros((grid_points_number-1, model.wavelengths.size))

#     spectrum = model.calcSpectrum(
#         surface_gravity, 
#         planet_radius, 
#         radius_ratio, 
#         pressure, 
#         temperature, 
#         chem_species, 
#         mix_ratios,
#         cloud_optical_depth)
    
#     # if FWHM_vals is not None:
#     #     interp_FWHM_vals = np.interp(model.wavelengths, FWHM_vals[:, 0], FWHM_vals[:, 1])
#     #     spectrum = model.convolve_spectrum(spectrum, interp_FWHM_vals)

#     return spectrum

def retrieval_forward_model(x, model, grid_points_number, chem_species, FWHM_vals=None):
    surface_gravity = 10**x[0]
    planet_radius = x[1] * const.R_earth.cgs.value
    radius_ratio = planet_radius / (x[2] * const.R_sun.cgs.value)

    #p-T structure
    temperature = np.full((grid_points_number), x[10])
    pressure = np.logspace(1, -7, grid_points_number)
    #chemical composition
    mix_ratios = np.zeros((grid_points_number, chem_species.size))  # also includes H2 and He, which will be set later
    total_metals = np.sum(10**x[3:10])

    # skip unphysical models
    if total_metals >= 1.0:
        return np.zeros(model.wavelengths.size)
    
    H2He_ratio = 0.83/0.13
    mix_ratios[:, 0] = (1.0 - total_metals) * H2He_ratio / (1.0 + H2He_ratio)
    mix_ratios[:, 1] = (1.0 - total_metals) / (1.0 + H2He_ratio)
    mix_ratios[:, 2:] = 10**x[3:10]

    print(mix_ratios[0,:], np.sum(mix_ratios[0,:]))
    #cloud properties, set to zero for now
    cloud_optical_depth = np.zeros((grid_points_number-1, model.wavelengths.size))

    print(f"Evaluating model with surface gravity={surface_gravity:.2e} cm/s^2, \n\
            planet radius={planet_radius:.2e} cm, \n\
            radius ratio={radius_ratio:.4f}, \n\
            temperature={temperature[0]:.1f} K, \n\
            total metals={total_metals:.2e}, \n\
            RV shift={x[13]:.1f} km/s\n\
            pressure range={pressure.min():.2e} - {pressure.max():.2e} bar\n\
            cloud optical depth range={cloud_optical_depth.min():.2e} - {cloud_optical_depth.max():.2e}")
    spectrum = model.calcSpectrum(
        surface_gravity, 
        planet_radius, 
        radius_ratio, 
        pressure, 
        temperature, 
        chem_species, 
        mix_ratios,
        cloud_optical_depth,
        use_variable_gravity=False)
    
    print('Model evaluated with parameters:', x)
    print(np.nanmean(spectrum), np.nanmin(spectrum), np.nanmax(spectrum))
    
    shifted_wavelengths = RV_shift_model(model.wavelengths, x[13])
    
    
    # if FWHM_vals is not None:
    #     interp_FWHM_vals = np.interp(model.wavelengths, FWHM_vals[:, 0], FWHM_vals[:, 1])
    #     spectrum = model.convolve_spectrum(spectrum, interp_FWHM_vals)

    return shifted_wavelengths, spectrum

def bin_to_data(wavelengths, spectrum, data_wavelengths):
    """Bin the model spectrum to the data wavelength grid."""
    binned_spectrum = np.zeros(data_wavelengths.size)
    for i in range(data_wavelengths.size):
        # find the indices of the model wavelengths that fall within the data bin
        if i == 0:
            bin_min = data_wavelengths[i] - (data_wavelengths[i+1] - data_wavelengths[i]) / 2.
        else:
            bin_min = (data_wavelengths[i-1] + data_wavelengths[i]) / 2.
        if i == data_wavelengths.size - 1:
            bin_max = data_wavelengths[i] + (data_wavelengths[i] - data_wavelengths[i-1]) / 2.
        else:
            bin_max = (data_wavelengths[i] + data_wavelengths[i+1]) / 2.
        
        in_bin = (wavelengths >= bin_min) & (wavelengths < bin_max)
        if np.any(in_bin):
            binned_spectrum[i] = np.nanmean(spectrum[in_bin])
        else:
            binned_spectrum[i] = np.nan  # or some other placeholder for empty bins
    return binned_spectrum

# def loglike(x):
#     """Gaussian loglikelihood"""
#     return -0.5 * np.dot(x, x) + lnorm


def priortransform(u):
    """Transforms the uniform random variables `u ~ Unif[0., 1.)`
    to the parameters of interest."""
    x = np.array(u)  # copy u

    # gaussians
    t = norm.ppf(u[0:3:2])
    x[0] = t[0] * 0.046 + 3.0126
    x[2] = t[1] * 0.011 + 0.378

    # planet radius [1.8, 2.5]
    x[1] = u[1] * (2.5-1.8) + 1.8

    # log-uniform species [1e-12, 1.0]
    x[3:10] = (u[3:10]-1) * 12.0

    # temperature [100, 800]
    x[10] = u[10] * (800.-100.) + 100.

    # shifts [-100, 100]
    x[11] = u[11] * (100.+100.) - 100.
    x[12] = u[12] * (100.+100.) - 100.

    # rv_shift [-200, 200] km/s
    x[13] = u[13] * (200.+200.) - 200.

    return x


if __name__ == "__main__":
    
    # transmission_model, chem_species, grid_points_number = setup_retrieval_model()

    # # define the planet/atmosphere parameters
    # surface_gravity = 1050  # in cm/s^2
    # planet_radius = 2.15  # in Earth radii
    # star_radius = 0.378 # in Solar radii
    # temperature = 380.0  # in K
    # p_bottom = 10.0  # in log10(bar)
    # p_top = -7.0  # in log10(bar)
    # mixing_ratios = np.array([1e-2, 2e-2, 3e-2, 1e-6, 5e-4, 1e-3, 1e-5])  # for H2O, CH4, CO2, CO, NH3, CS2, SO2

    # params = params(surface_gravity, planet_radius, star_radius, temperature, p_bottom, p_top, mixing_ratios)

    # spectrum = forward_model(params, transmission_model, grid_points_number, chem_species)

    # fig, ax = plt.subplots()
    # ax.plot(transmission_model.wavelengths, spectrum)
    # plt.xlabel("Wavelength ($\mu$m)")
    # plt.ylabel("Transit depth (ppm)")
    # plt.savefig("/work2/lbuc/lukas/Projects/CC_JWST/figures/retrieval_test_spectrum.png", dpi=300)
    # # plt.show()
    # plt.clf()

    transmission_model, chem_species, grid_points_number = setup_retrieval_model()
    
    niriss = np.loadtxt('/work2/lbuc/lukas/data/TOI-270/fullres_NIRISS_quadratic_ExoCTK_5x.dat', skiprows=11)
    wls_niriss, depths_niriss, errs_niriss, FWHM_niriss = niriss[:,0], niriss[:,1], niriss[:,2], niriss[:, 3]
    # niriss = np.loadtxt('/work2/lbuc/lukas/data/TOI-270/fullres_NIRISS2......dat', skiprows=11)
    nrs1 = np.loadtxt('/work2/lbuc/lukas/data/TOI-270/fullres_NRS1_quadratic_ExoCTK_5x-wide_flat_d.dat', skiprows=11)
    wls_nrs1, depths_nrs1, errs_nrs1, FWHM_nrs1 = nrs1[:,0], nrs1[:,1], nrs1[:,2], nrs1[:, 3]
    nrs2 = np.loadtxt('/work2/lbuc/lukas/data/TOI-270/fullres_NRS2_quadratic_ExoCTK_5x-wide_flat_d.dat', skiprows=11)
    wls_nrs2, depths_nrs2, errs_nrs2, FWHM_nrs2 = nrs2[:,0], nrs2[:,1], nrs2[:,2], nrs2[:, 3]

    datasets = [
        '/work2/lbuc/lukas/data/TOI-270/fullres_NIRISS_quadratic_ExoCTK_5x.dat',
        '/work2/lbuc/lukas/data/TOI-270/fullres_NRS1_quadratic_ExoCTK_5x-wide_flat_d.dat',
        '/work2/lbuc/lukas/data/TOI-270/fullres_NRS2_quadratic_ExoCTK_5x-wide_flat_d.dat'
    ]

    def loglikelihood(theta):
        """
        Gaussian log-likelihood for multiple datasets with per-dataset offsets.
        
        Parameters (from priortransform):
        theta[0]    : log10(surface_gravity)
        theta[1]    : planet_radius [R_earth]
        theta[2]    : star_radius [R_sun]
        theta[3-9]  : log10(mixing ratios) for H2O, CH4, CO2, CO, NH3, CS2, SO2
        theta[10]   : temperature [K]
        theta[11]   : offset_NIRISS  [ppm]
        theta[12]   : offset_NRS2    [ppm]  -- NRS1 offset is the reference (=0) or also free
        theta[13]   : rv_shift [km/s]
    """

        # Evaluate model at each data point
        wls, model = retrieval_forward_model(theta, transmission_model, grid_points_number, chem_species)
        
        # If model is unphysical (total_metals >= 1), forward model returns zeros
        if np.all(model == 0.0):
            return -np.inf
        
        # Residuals
        model_convolved_niriss = convolve_spectrum(wls, model, wls_niriss, FWHM_niriss)
        y_niriss = bin_to_data(wls, model_convolved_niriss, wls_niriss) + theta[11]  # add NIRISS offset

        model_convolved_nrs1 = convolve_spectrum(wls, model, wls_nrs1, FWHM_nrs1)
        y_nrs1 = bin_to_data(wls, model_convolved_nrs1, wls_nrs1)  # no NRS1 offset

        model_convolved_nrs2 = convolve_spectrum(wls, model, wls_nrs2, FWHM_nrs2)
        y_nrs2 = bin_to_data(wls, model_convolved_nrs2, wls_nrs2) + theta[12]  # add NRS2 offset

        residual_niriss = depths_niriss - y_niriss
        residual_nrs1 = depths_nrs1 - y_nrs1
        residual_nrs2 = depths_nrs2 - y_nrs2

        # Log-likelihood (constant log(2*pi) term often dropped, but included here for correctness)
        loglike = -0.5 * np.sum((residual_niriss / errs_niriss) ** 2 + np.log(2 * np.pi * errs_niriss**2)) + \
                  -0.5 * np.sum((residual_nrs1 / errs_nrs1) ** 2 + np.log(2 * np.pi * errs_nrs1**2)) + \
                  -0.5 * np.sum((residual_nrs2 / errs_nrs2) ** 2 + np.log(2 * np.pi * errs_nrs2**2))

        return loglike
    
    def forward_data(theta):
        """
        Gaussian log-likelihood for multiple datasets with per-dataset offsets.
        
        Parameters (from priortransform):
        theta[0]    : log10(surface_gravity)
        theta[1]    : planet_radius [R_earth]
        theta[2]    : star_radius [R_sun]
        theta[3-9]  : log10(mixing ratios) for H2O, CH4, CO2, CO, NH3, CS2, SO2
        theta[10]   : temperature [K]
        theta[11]   : offset_NIRISS  [ppm]
        theta[12]   : offset_NRS2    [ppm]  -- NRS1 offset is the reference (=0) or also free
        theta[13]   : rv_shift [km/s]
    """

        # Evaluate model at each data point
        print("Evaluating forward model with parameters:")
        print(theta)
        wls, model = retrieval_forward_model(theta, transmission_model, grid_points_number, chem_species)
        
        # Residuals
        print("Calculating model for each instrument with convolution and binning...")
        model_convolved_niriss = convolve_spectrum(wls, model, wls_niriss, FWHM_niriss)
        # model_convolved_niriss = model
        y_niriss = bin_to_data(wls, model_convolved_niriss, wls_niriss) + theta[11]  # add NIRISS offset

        model_convolved_nrs1 = convolve_spectrum(wls, model, wls_nrs1, FWHM_nrs1)
        # model_convolved_nrs1 = model
        y_nrs1 = bin_to_data(wls, model_convolved_nrs1, wls_nrs1)  # no NRS1 offset

        model_convolved_nrs2 = convolve_spectrum(wls, model, wls_nrs2, FWHM_nrs2)
        # model_convolved_nrs2 = model
        y_nrs2 = bin_to_data(wls, model_convolved_nrs2, wls_nrs2) + theta[12]  # add NRS2 offset

        print("Done. Average model values:")
        print(np.nanmean(y_niriss), np.nanmean(y_nrs1), np.nanmean(y_nrs2))
        return y_niriss, y_nrs1, y_nrs2
    
    plt.figure(figsize=(10, 6))
    test_theta = np.array([3.0126, 2.15, 0.378, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 450.0, 20.0, 0.0, 0.0])  # example parameters for testing
    # y_niriss, y_nrs1, y_nrs2 = forward_data(priortransform(np.random.rand(14)))
    wls, model = retrieval_forward_model(test_theta, transmission_model, grid_points_number, chem_species)
    print("Model wavelengths range:", wls.min(), wls.max())
    print("Model transit depth range:", model.min(), model.max())
    # plt.plot(wls_niriss, y_niriss, label='NIRISS')
    # plt.plot(wls_nrs1, y_nrs1, label='NRS1')
    # plt.plot(wls_nrs2, y_nrs2, label='NRS2')
    plt.plot(wls, model, label='Model spectrum')
    plt.xlabel("Wavelength ($\mu$m)")
    plt.ylabel("Transit depth (ppm)")
    plt.legend()
    plt.savefig("/work2/lbuc/lukas/Projects/CC_JWST/figures/retrieval_test_spectrum.png", dpi=300)
    plt.clf()


    # # "Static" nested sampling.
    # sampler = dynesty.NestedSampler(loglikelihood, priortransform, ndim, nlive=500)
    # sampler.run_nested(dlogz=0.9, checkpoint_file='dynesty.save')
    # sresults = sampler.results
    # sresults.summary()
    # equal_results = sresults.samples_equal()

    # outfile = open('/work2/lbuc/lukas/Projects/CC_JWST/retrievals/results/test.save', 'ab')
    # pickle.dump(sresults, outfile)
    # outfile.close()
    # new_outfile = open('/work2/lbuc/lukas/Projects/CC_JWST/retrievals/results/test_equal.save', 'ab')
    # pickle.dump(equal_results, new_outfile)
    # new_outfile.close()
    # rfig, axes = dyplot.runplot(sresults)
    # plt.savefig("/work2/lbuc/lukas/Projects/CC_JWST/figures/retrieval_test_spectrum.png", dpi=300)