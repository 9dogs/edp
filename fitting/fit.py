import numpy as np
import pandas as pd

from collections import OrderedDict

from lmfit.models import LorentzianModel, ConstantModel, Model, GaussianModel, VoigtModel
from lmfit.parameter import Parameters


PEAK_MODELS = OrderedDict({
    'Gauss': GaussianModel,
    'Lorentz': LorentzianModel,
    'Voigt': VoigtModel
})


def echo_decay_curve(x, y0, A, t2):
    return y0 + A*np.exp(-4 * x / t2)


def prepare_data(df, laser_window, laser_title='laser', time_title='time'):
    """
    Loads echo data from `filename` and filter with laser_window = [min, max] ([mean-std, mean+std] on default)
    """
    df_orig = df
    df_orig.columns = [c.lower().replace(' ', '_') for c in df_orig.columns]

    l_min, l_max = laser_window

    df = df_orig[(df_orig[laser_title] < l_max) & (df_orig[laser_title] > l_min)]

    df_mean = df.groupby(time_title).mean()
    df_mean.columns = [c + "_mean" for c in df_mean.columns]

    df_std = df.groupby(time_title).std()
    df_std.columns = [c + "_std" for c in df_std.columns]

    return pd.concat([df_mean, df_std], axis=1)


def fit_peak_df(df, model=GaussianModel, params=None, fit_range=(-np.inf, np.inf), x_field=None, fit_field='nphe2_mean', out_field='peak_fit'):
    """
    Fits DataFrame with selected peak model. Appends residuals column to DataFrame.
    """
    # Data
    fit_min, fit_max = fit_range

    df_ranged = df[(df.index > fit_min) & (df.index < fit_max)]

    if x_field:
        x = np.array(df_ranged[x_field])
        full_x = np.array(df[x_field])
    else:
        x = np.array(df_ranged.index.get_values())
        full_x = np.array(df.index.get_values())

    y = np.array(df_ranged[fit_field].values)
    full_y = np.array(df[fit_field].values)

    # Models
    if isinstance(model, str):
        try:
            model = PEAK_MODELS[model]
        except KeyError:
            print("Undefined model: {}, using default".format(model))

    peak_mod = model(prefix='peak_')
    const_mod = ConstantModel(prefix='const_')
    result_model = const_mod + peak_mod

    # Parameters
    if not params:
        pars = const_mod.make_params(c=y.min())
        pars += peak_mod.guess(y, x=x, center=0)
    else:
        pars = params

    # Fitting
    result = result_model.fit(y, params=pars, x=x)

    peak_eval = result.eval(x=full_x)
    y_res = full_y - peak_eval
    df[out_field] = pd.Series(peak_eval, index=df.index)
    df[out_field + '_res'] = pd.Series(y_res, index=df.index)

    return df, result


def _init_echo_params(y0, A, t2):
    """
    Params for echo model
    """
    params = Parameters()
    y0_min = None
    y0_max = None
    y0_value = y0
    y0_vary = True
    params.add('y0', min=y0_min, max=y0_max, value=y0_value, vary=y0_vary)

    A_min = None
    A_max = None
    A_value = A
    A_vary = True
    params.add('A', min=A_min, max=A_max, value=A_value, vary=A_vary)

    t2_min = None
    t2_max = 10000
    t2_value = t2
    t2_vary = True
    params.add('t2', min=t2_min, max=t2_max, value=t2_value, vary=t2_vary)

    return params


def fit_residuals(df, start_fit_from=0.4, fit_field='fit_lorenz_res', init_params=None):
    """
    Fits residuals with EchoDecay curve
    """
    x = np.array(df[df.index > start_fit_from].index.get_values())
    y = np.array(df[df.index > start_fit_from][fit_field].values)

    echo_model = Model(echo_decay_curve, independent_vars=['x'])
    if init_params:
        params = init_params
    else:
        params = _init_echo_params(0, 500, 50)
    result = echo_model.fit(y, x=x, params=params)

    full_x = np.array(df.index.get_values())
    res_fit_eval = result.eval(x=full_x)
    df['res_echo_fit'] = pd.Series(res_fit_eval, index=df.index)

    return df, result
