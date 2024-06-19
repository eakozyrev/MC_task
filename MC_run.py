import numpy as np
import matplotlib.pyplot as plt
import math
import array as array
import scipy
from scipy import stats
import argparse


class RQuant_Stable:
    '''Class to generate return1 and return10 spectra and its quantiles'''

    def __init__(self,ndays = 750,step = 10,quant_method = 'hazen'):
        '''
        self.ndays -- number of return1 values
        self.quant_method -- the method to get quantile, by default = 'hazen'
        self.step -- number of days in the higher order return^k, by default = 10
        self.data1 -- np.array of return1
        self.data10 -- np.array of return^k
        '''
        self.ndays = ndays
        self.quant_method = quant_method
        self.step = step
        self.data1 = np.zeros(self.ndays)
        self.data10 = np.zeros(self.ndays-self.step + 1)

    def gen_stable(self):
        '''
        this method is not used. It is created for validation on scipy.stats.levy_stable \n
        [http://prac.im.pwr.edu.pl/~hugo/RePEc/wuu/wpaper/HSC_10_05.pdf] 
        '''
        data = np.zeros(self.ndays)
        for i in range(self.ndays):
            U = np.pi*np.random.random_sample() - np.pi/2.
            W = np.random.exponential(1)
            X = math.sin(1.7*U)/(math.cos(U))**(1./1.7)*(math.cos(U-1.7*U)/W)**(-0.7/1.7)
            Y = X + 1.
            data[i] = Y
        return data
    
    def gen_stable_scipy(self):
        '''
        run scipy generation of stable distribution with (alpha=1.7,beta=0,loc=1,scale=1);
        self.data1 will be filled by ndays return1 values;
        :return: None.
        '''
        gen = scipy.stats.levy_stable(alpha=1.7,beta=0,loc=1,scale=1)
        self.data1 = gen.rvs(size = self.ndays)


    def get_n_from_stable(self):
        """
        The method generate return^k spectrum and fill self.data10;
        :return: None.
        """
        retn = self.data1[0]
        for i in range(1,self.step):
            retn = retn*(1+self.data1[i])
        self.data10[0] = retn
        for i in range(1,self.data1.size-self.step+1):
            retn = retn/self.data1[i-1]/(1/self.data1[i]+1)*(1+self.data1[i+self.step-1])
            self.data10[i] = retn   
    
    def fill_single(self,a,quantile):
        """
        Generate stable spectrum, distribute return10 and calculate quantile;
        :param a: input data to calculate quantile;
        :param quantile: the value of level of required quantile;
        :return: quantile.
        """
        self.gen_stable_scipy()
        self.get_n_from_stable()
        return np.quantile(a=self.data10,q=quantile,method='hazen')

    def get_n_quantile_p(self,nevents,quantile = 0.01):
        """
        Generate the spectrum of quantiles with nevents of events;
        :param quantile: the required level of quantile, by default = 0.01;
        :return: np.array of spectrum.
        """
        vectorized_fill_single = np.vectorize(self.fill_single)
        res = np.zeros(shape=nevents)
        return vectorized_fill_single(res, quantile)
    
    def get_n_quantile_p_while(self,quantile = 0.01,delta = 0.01):
        """
        Generate the spectrum of quantiles untill the mean of the spectrum fluctuates;
        :param quantile: the value of quantile level, by default = 0.01;
        :param delta: generate spectrum untill mean[-10:].std() < |delta*mean|;
        return: res - the required spectrum of quantiles.
        """
        res = np.array([])
        data_mean = np.array([])
        while True:
            data_stable = self.gen_stable_scipy()
            data_stable_10 = self.get_n_from_stable()
            quant = np.quantile(a=self.data10,q=quantile,method='hazen')
            res = np.append(res,quant)
            data_mean = np.append(data_mean,res.mean())
            if res.shape[0] > 10 and data_mean[-10:].std() < np.fabs(delta*res.mean()): break
        return res


def plot_histogram(ax, data, bins, range, title, log=False, histtype='bar', label=None, density=False):
    """
    plot histogramm
    """
    ax.hist(data, bins=bins, range=range, histtype=histtype, log=log, density=density, label=label)
    ax.set_title(title)
    if label:
        ax.legend()

def main(nevent, delta):
    """
    :param nevent: generate 1st spectrum with nevents
    :param delta: generate 2nd spectrum untill mean[-10:].std() < |delta*mean|;
    """
    generator = RQuant_Stable(ndays=750)
    np.random.seed(123)
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))

    # generate and plot return1 spectrum
    data_stable = generator.gen_stable_scipy()
    data1_range = [np.quantile(generator.data1, 0.01), np.quantile(generator.data1, 0.99)]
    plot_histogram(ax[0, 0], generator.data1, bins=100, range=data1_range, title='stable distribution of return1', histtype='step')

    # generate and plot return10 spectrum
    data_10 = generator.get_n_from_stable()
    data10_range = [np.quantile(generator.data10, 0.01), np.quantile(generator.data10, 0.99)]
    plot_histogram(ax[0, 1], generator.data10, bins=500, range=data10_range, title='distribution of return10', log=True)

    # generate high statistics spectrum of quantiles
    data_q_limit = generator.get_n_quantile_p(nevents=nevent, quantile=0.01)
    q_limit_range = (data_q_limit.min(), data_q_limit.max())
    plot_histogram(ax[1, 0], data_q_limit, bins=100, range=q_limit_range, title='1% quantile spectrum in [min,max] range', log=True, density=True, label=f'n events = {nevent}')

    # generate small sample of events
    data_q = generator.get_n_quantile_p_while(quantile=0.01,delta=delta)
    plot_histogram(ax[1, 0], data_q, bins=100, range=q_limit_range, title='', log=True, histtype='step', density=True, label=f'n events = {data_q.shape[0]}')

    q = np.quantile(data_q_limit, 0.01, method='hazen')
    plot_histogram(ax[1, 1], data_q_limit, bins=100, range=(q, data_q_limit.max()), title='1% quantile spectrum in [1% quantile,max] range', log=True, density=True, label=f'n events = {nevent}')
    plot_histogram(ax[1, 1], data_q, bins=100, range=(q, data_q_limit.max()), title='', log=True, histtype='step', density=True, label=f'n events = {data_q.shape[0]}')

    print(f'min/max of generated spectrum: {np.round(data_q.min())} {np.round(data_q.max())}')
    print(f'Number of events = {data_q.shape[0]}')
    print('Kolmogorov-Smirnov test comparing generated spectrum and high statistics spectrum')
    print(stats.ks_2samp(data_q, data_q_limit))
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--observations_num', default=4000, required=False, type=int,
                        help='number of generate quantiles')
    parser.add_argument('-d', '--stop_generate_at', default=0.01, required=False, type=float,
                        help='the sign to stop generation')
    args = parser.parse_args()

    distribution_params = args.observations_num, args.stop_generate_at
    main(distribution_params[0],distribution_params[1])