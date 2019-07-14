import numpy as np
import pandas as pd
import scipy.stats as st
import scipy
import statsmodels as sm
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

class FitDistributions:

    def __init__(self, type = 'continuous', plot_all = False):
        warnings.filterwarnings('ignore')
        self.type = type
        self.plot_all = plot_all
        self.results = {}

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

    def fit_distribution(self, data, bins=None):
        self.bins = bins
        if self.bins is None:
            self.bins = np.histogram_bin_edges(data)
       
        if self.type == 'continuous':
            y, x = np.histogram(data, bins=self.bins, density=True)
            x = (x + np.roll(x, -1))[:-1] / 2.0

            best_distribution = None
            best_params = None
            best_sse = np.inf

            for d in tqdm(self.distributions):
                try:
                    params = d.fit(data)
                    pdf = d.pdf(
                        x, loc=params[-2], scale=params[-1], *params[:-2])
                    sse = np.sum((y - pdf)**2)
                    self.results[d.name] = {'params': params, 'sse': sse}

                    if best_sse > sse > 0:
                        best_distribution = d
                        best_params = params
                        best_sse = sse

                except Exception as e:
                    continue

            self._plot(data, best_distribution.name, best_params)
            return best_distribution.name, best_params

        if self.type == 'discrete':
            return data / np.sum(data)

    def _plot(self, data, dist_name, params, size=10000):
        plt.hist(data, normed=1, bins=self.bins)
        x = np.linspace(np.min(data), np.max(data), size)
        plt.title('Fitted Data')

        if self.plot_all:
            for dist in self.distributions:
                params = self.results[dist.name]['params']
                y = dist.pdf(x, loc=params[-2], scale=params[-1], *params[:-2])
                plt.plot(x, y, label=dist.name)
                del y
        else:
            dist = getattr(st, dist_name)
            y = dist.pdf(x, loc=params[-2], scale=params[-1], *params[:-2])
            plt.plot(x, y, label=dist.name)
        
        plt.legend(loc='upper right')
        plt.show()

    def generate_rvs(self, size, dist_name, params):
        d = getattr(st, dist_name)
        return d.rvs(loc=params[-2], scale=params[-1], *params[:-2], size = size)

if __name__ == '__main__':
    data = st.norm.rvs(1, 2, size=5000)
    DF = FitDistributions(plot_all = True)
    DF.distributions = [st.norm, st.maxwell, st.uniform]
    dist_name, params = DF.fit_distribution(data, bins = 100)
    print(dist_name, params)