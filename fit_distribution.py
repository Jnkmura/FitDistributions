import numpy as np
import pandas as pd
import scipy.stats as st
import scipy
import statsmodels as sm
import matplotlib.pyplot as plt
from tqdm import tqdm

class FitDistributions:

    def __init__(self, type = 'continuous'):
        self.type = type
        if self.type == 'continuous':
            self.distributions = [        
                    st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
                    st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
                    st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
                    st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
                    st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
                    st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
                    st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
                    st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
                    st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
                    st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
                ]

    def fit_distribution(self, data, bins=200):
        self.bins = bins
        if self.type == 'continuous':
            y, x = np.histogram(data, bins=self.bins, density=True)
            x = (x + np.roll(x, -1))[:-1] / 2.0

            best_distribution = None
            best_params = None
            best_sse = np.inf

            for d in tqdm(self.distributions):
                try:
                    params = d.fit(data)
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    pdf = d.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))

                    if best_sse > sse > 0:
                        best_distribution = d
                        best_params = params
                        best_sse = sse

                except Exception as e:
                    continue

            self._plot(x, data, best_distribution.name, best_params)
            return best_distribution.name, best_params

        if self.type == 'discrete':
            return data / np.sum(data)

    def _plot(self, x, data, dist_name, params):
        dist = getattr(st, dist_name)
        pdf = dist.pdf(
            x, loc=params[-2], scale=params[-1], *params[:-2])
        plt.hist(data, bins=self.bins, normed=1)
        plt.plot(pdf, label=dist_name)
        plt.show()

data = np.random.normal(1000, 500, 5000)
DF = FitDistributions()
a, b = DF.fit_distribution(data)

#https://stackoverflow.com/a/37616966