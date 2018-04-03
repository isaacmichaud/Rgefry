import pymc
import matplotlib.pyplot as plt
import gefry3
import pickle as pickle
import numpy as np
from shapely import geometry as g
from shapely.ops import cascaded_union
from scipy.stats import multivariate_normal
import yaml

def run_mcmc(deck,outfile,n_samples = 1e5):
    dA = 0.005806  # m^2
    my_deck    = file(deck)
    gefry_deck = yaml.safe_load(my_deck)

    P = gefry3.read_input_problem(
        deck,
        problem_type="Perturbable_XS_Problem"
    )

    NS = n_samples

    S0 = P.source.R # m
    I0 = P.source.I0 # Bq
    BG = 300 # cps

    XMIN, YMIN, XMAX, YMAX = P.domain.all.bounds
    IMIN, IMAX = 1e9, 5e9
    #IMIN, IMAX = 1e8, 1e10

    # Relative perturbation used for all cross sections
    XS_DELTA = 0.5

    # Generate some data

    DWELL = np.array([i.dwell for i in P.detectors])

    obs   = [i for i in gefry_deck["observations"]]
    #print(P(S0, I0, inter_mat, building_mats)

    # Call P at the nominal values to get the real response
    nominal = P(S0, I0, P.interstitial_material, P.materials)
    print(nominal)

    #nominal += BG * DWELL

    # Generate the data and the covariance assuming detectors are independent
    # (a pretty safe assumption).
    #data = np.random.poisson(nominal)

    data   = obs
    C_data = np.diag(data)

    def model_factory():
        """Build a PyMC model and return it as a dict"""

        x = pymc.Uniform("x", value=S0[0], lower=XMIN, upper=XMAX)
        y = pymc.Uniform("y", value=S0[1], lower=YMIN, upper=YMAX)
        I = pymc.Uniform("I", value=I0, lower=IMIN, upper=IMAX)

        # Distributions for the cross sections

        # Just the interstitial material
        s_i_xs = P.interstitial_material.Sigma_T
        interstitial_xs = pymc.Uniform(
            "Sigma_inter",
            s_i_xs * (1 - XS_DELTA),
            s_i_xs * (1 + XS_DELTA),
            value=s_i_xs,
            observed=False,
        )

        # All the rest
        mu_xs = np.array([M.Sigma_T for M in P.materials])

        building_xs = pymc.Uniform(
            "Sigma",
            mu_xs * (1 - XS_DELTA),
            mu_xs * (1 + XS_DELTA),
            value=mu_xs,
            observed=False,
        )

        # Predictions

        @pymc.deterministic(plot=False)
        def model_pred(x=x, y=y, I=I, interstitial_xs_p=interstitial_xs, building_xs_p=building_xs):
            # The _p annotation is so that I can access the actual stochastics
            # in the enclosing scope, see down a couple lines where I resample

            inter_mat = gefry3.Material(1.0, interstitial_xs_p)
            building_mats = [gefry3.Material(1.0, s) for s in building_xs_p]

            # Force the cross sections to be resampled
            interstitial_xs.set_value(interstitial_xs.random(), force=True)
            building_xs.set_value(building_xs.random(), force=True)

            return P(
                [x, y],
                I,
                inter_mat,
                building_mats,
            )

        background = pymc.Poisson(
            "b",
            DWELL * BG,
            value=DWELL * BG,
            observed=False,
            plot=False,
        )

        @pymc.stochastic(plot=False, observed=True)
        def observed_response(value=[], model_pred=model_pred, background=background):
            resp = model_pred + background
            return multivariate_normal.logpdf(data, mean=resp, cov=np.diag(resp))

        return {

            "x": x,
            "y": y,
            "I": I,
            "interstitial_xs": interstitial_xs,
            "building_xs": building_xs,
            "model_pred": model_pred,
            "background": background,
            "observed_response": observed_response,
        }

    # Set up the sampler and run
    mvars = model_factory()

    M = pymc.MCMC(mvars)

    M.use_step_method(
        pymc.AdaptiveMetropolis,
        [mvars[i] for i in mvars]
    )

    M.sample(NS)
    pymc.utils.coda(M,name=outfile)

#run_mcmc("test_deck.yml","test_coda",5000)
