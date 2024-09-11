import os
import sys
import argparse
import numpy as np
import logging 
from math import pow
from functools import wraps
from scipy.stats import binom, norm
from utils_v01 import *


def create_logger():
    logger_obj = logging.getLogger(__name__)
    logger_obj.setLevel(logging.DEBUG)
    log_filename = os.path.splitext(os.path.basename(__file__))[0] + "_py.log"
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'logs')): os.makedirs(os.path.join(os.path.dirname(__file__), 'logs'))
    #create the handlers for the logger
    console_handler= logging.StreamHandler(stream=sys.stdout)
    file_handler = logging.FileHandler(os.path.join(os.path.dirname(__file__), 'logs', log_filename), mode= 'a')
    #create formatter and assign to handlers
    log_formatter=logging.Formatter('[%(asctime)s - %(name)s - %(lineno)d - %(levelname)s ]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S', style='%')
    console_handler.setFormatter(log_formatter)
    file_handler.setFormatter(log_formatter)
    #Add the handlers to the logger object
    logger_obj.addHandler(console_handler)
    logger_obj.addHandler(file_handler)

    return logger_obj

def get_cmdline_args():
    parser = argparse.ArgumentParser(description= 'Arguments for Option Contract Valuation')
    parser.add_argument('--spot_price', dest= 'spot_price', type= float, help = 'price of stock at time zero ($  per share)')
    parser.add_argument('--strike_price', dest= 'strike_price', type= float, help = 'Striek or exercise price of the udnerlying asset($  per share)')
    parser.add_argument('--riskfree_rate', dest= 'riskfree_rate', type= float, help = 'risk-free interest rate')
    parser.add_argument('--expiry_time', dest= 'expiry_time', type= float, help = 'Time-to-expiration of option contract (expressed in years)')
    parser.add_argument('--volatility', dest= 'volatility', type= float, help = 'volatility of option contract (%p.a)')
    parser.add_argument('--dividend_yield', dest= 'dividend_yield', type= float, help = 'Dividend-yield of option contract (%p.a)')
    parser.add_argument('--option_type', dest= 'option_type', type= str, help = 'Type of option contract (%p.a)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-q', '--quiet', dest= 'quiet', help='print little output', action = 'store_true')
    group.add_argument('-v', '--verbose', dest= 'verbose', help= 'print extra output', action= 'store_true')

    return parser.parse_args()


class QriskNeutralProbError(Exception):
    default_msg = "Error related to risk neurtral probabilities"
    def __init__(self, msg= None):
        self.msg = msg
        super().__init__(msg or default_msg)

class BS_NormalCumDistError(Exception):
    default_msg = "Error related to Black-Scholes Normal Cumulative Distribution of d1 & d2"
    def __init__(self, msg= None):
        self.msg= msg
        super().__init__(msg, default_msg)

class OptionContractPricing:
    def __init__(self, spot_price, strike_price, riskfree_rate, expiry_time, volatility, dividend_yield, option_type=('Call', 'Put'), logger= None):
        self._spot_price = spot_price
        self._strike_price = strike_price
        self._riskfree_rate = riskfree_rate
        self._expiry_time = expiry_time
        self._volatility = volatility
        self.option_type= option_type
        self._dividend_yield = dividend_yield
        self.logger = logger

    @property
    def spot_price(self):
        return self._spot_price

    @property
    def strike_price(self):
        return self._strike_price

    @property
    def riskfree_rate(self):
        return self._riskfree_rate
    
    @property
    def expiry_time(self):
        return self._expiry_time

    @property
    def volatility(self):
        return self._volatility

    @property
    def dividend_yield(self):
        return self._dividend_yield

    @spot_price.setter
    def spot_price(self, v):
        if not isinstance(v, (float, int)):
            raise TypeError("Data type of 'spot-price' is neither a float or integer")
        if v < 0:
            raise ValueError("The stock-price cannot be negative or less than $0.00")
        self._spot_price = v

    
    @strike_price.setter
    def strike_price(self, v):
        if not isinstance(v, (float, int)):
            raise TypeError("Data type of 'strike-price' is neither a float or integer")
        if v < 0:
            raise ValueError("The strike-price cannot be negative or less than $0.00")
        self._strike_price = v

    @riskfree_rate.setter
    def riskfree_rate(self, v):
        if not isinstance(v, (float, int)):
            raise TypeError("Data type of 'riskfree-interest-rate' is neither a float or integer")
        self._riskfree_rate = v

    @dividend_yield.setter
    def dividend_yield(self, v):
        if not isinstance(v, (float, int)):
            raise TypeError("Data type of 'dividend-yield' is neither a float or integer")
        self._dividend_yield = v

    @expiry_time.setter
    def expiry_time(self, v):
        if not isinstance(v, (float, int)):
            raise TypeError("Data type of Expiry-Time (Exercise TIme) is no an integer")
        if v < 0:
            raise ValueError("The expiry-time cannot be negative.")
        self._expiry_time = v

    @volatility.setter
    def volatility(self, v):
        if not isinstance(v, (float, int)):
            raise TypeError("Data type of volatility is not an integer")
        if v < 0:
            raise ValueError("Volatility cannot be negative.")
        self._volatility = v

    def __repr__(self):
        return f"{__class__.__name__}({self._spot_price}, {self._strike_price}, {self._riskfree_rate}, {self._expiry_time},{self._volatility}, {self._dividend_yield}, {self.option_type})"

    def __str__(self):
        return str({"Spot Price" : "${:,.4f}".format(self._spot_price),
                    "Strike Price" : "${:,.4f}".format(self._strike_price),
                    "Riskfree Interest Rate" : self._riskfree_rate,
                    "Expiry (Exercise) Time" : self._expiry_time,
                    "Volatility" : self._volatility,
                    "Dividend-Yield" : self._dividend_yield,
                    "Option-Contract-Type" : self.option_type})

    def BinomialPricingModel(self, up_factor, down_factor):
        """ Option contract pricing or valudation using multi-period Binomial Pricing model."""
        if not isinstance(up_factor, (float, int)):
            raise TypeError("Data type of Up-factor is neiterh a float nor integer")
        if up_factor < 0:
            raise ValueError("Up factoe cannot be negative")
        if self.option_type not in ('Call' 'Put'):
            raise ValueError("Option type provided is not recongnized. It must be either Call or Put option contract")
        try:
            self._expiry_time = int(self._expiry_time)
            if down_factor=="None" or np.isnan(down_factor) : down_factor = 1/up_factor
            future_state_prices = [self._spot_price * pow(up_factor, self._expiry_time - k) * pow(down_factor, k) for k in range(int(self._expiry_time) + 1)]
            #compute Q-measure (i.e.risk neutral probabilities)
            q_prob_up = ((1 + self._riskfree_rate) - down_factor)/(up_factor - down_factor)
            discount_factor = 1/((1 + (self._riskfree_rate - self._dividend_yield))**self._expiry_time)
            #Q-meansure probabilities at n-th period state
            q_probs_nth_state = [binom.pmf(x, self._expiry_time, q_prob_up) for x in range(int(self._expiry_time) + 1)]
            for prob in q_probs_nth_state:
                if prob < 0 or prob > 1:
                    raise QriskNeutralProbError("Q-risk neutral probabilities out of bounds of [0,1] for option valuation")
            self.logger.info(f"Computed risk neutral probability (Q-measure) for up-factor : {q_prob_up}")
            self.logger.info(f"Computed risk neutral probability (Q-measure) for down-factor : {1 - q_prob_up}")
            #self.logger.info(f"Computed q-probabilities at {self._expiry_time}th period/time : {q_probs_nth_state}")
        except QriskNeutralProbError as err:
            self.logger.error(f"Error was thrown as : {err}", exc_info =True)
        else:
            self.logger.info(f"Computing the {self._expiry_time}th day period payoff of {self.option_type} option contract.")
            if self.option_type == 'Call':
                nth_period_payoff = [np.max([val - self._strike_price, 0], axis = 0) for val in future_state_prices]
                payoff_today = discount_factor * np.dot(q_probs_nth_state, nth_period_payoff)
                return payoff_today
            elif self.option_type == 'Put':
                nth_period_payoff = [np.max([self._strike_price - val, 0], axis= 0) for val in future_state_prices]
                payoff_today = discount_factor * np.dot(q_probs_nth_state, nth_period_payoff)
                return payoff_today
            else:
                self.logger.debug("Option type not recognized")

    def dplusCallOption(self):
        expiry_time_years = self._expiry_time/365
        dplus = (np.log(self._spot_price/self._strike_price) + (self._riskfree_rate + 0.5*self._volatility**2))/self._volatility*np.sqrt(expiry_time_years)
        dminus = dplus - self._volatility*np.sqrt(expiry_time_years)
        cdfNormal_dplus, cdfNormal_dminus = norm.cdf(dplus), norm.cdf(dminus)
        return dplus, dminus, cdfNormal_dplus, cdfNormal_dminus
                
    def dplusPutOption(self):
        dplus, dminus, cdfNormal_dplus, cdfNormal_dminus = self.dplusCallOption()
        cdfNormal_dplus, cdfNormal_dminus = norm.cdf(-1*dplus), norm.cdf(-1*dminus)
        return -1*dplus, -1*dminus, cdfNormal_dplus, cdfNormal_dminus 

    def BlackScholesModel(self):
        """ Option contract fair pricing or valuation using B-S Model with stochastic interest rates for divident paying stock"""

        if self.option_type not in ('Call', 'Put'):
            raise ValueError("Option value provided is not recognized. it must be either Call or Put option contract")
        try:
            expiry_time_years = self._expiry_time/255
            dplus, dminus, cdfNormal_dplus, cdfNormal_dminus = self.dplusCallOption()
            put_dplus, put_dminus, put_cdfNormal_dplus, put_cdfNormal_dminus = self.dplusCallOption()
            if cdfNormal_dplus < 0 or cdfNormal_dplus > 1:
                raise BS_NormalCumDistError("Normal Cumulative Distribution  Probability of d1 out of bounds of [0,1] for Call option valuation")
            if cdfNormal_dminus < 0 or cdfNormal_dminus > 1:
                raise BS_NormalCumDistError("Normal Cumulative Distribution  Probability of d1 out of bounds of [0,1] for Call option valuation")
            if put_cdfNormal_dplus < 0 or put_cdfNormal_dplus > 1:
                raise BS_NormalCumDistError("Normal Cumulative Distribution  Probability of d1 out of bounds of [0,1] for Put option valuation")
            if put_cdfNormal_dminus < 0 or put_cdfNormal_dminus > 1:
                raise BS_NormalCumDistError("Normal Cumulative Distribution  Probability of d1 out of bounds of [0,1] for Put option valuation")
        except BS_NormalCumDistError as err:
            self.logger.error("Error was thrown as : ", exc_info= True)
        else:
            if self.option_type == 'Call':
                return cdfNormal_dplus*self._spot_price - cdfNormal_dminus*self._strike_price * np.exp(-1*self._riskfree_rate*expiry_time_years)
            elif self.option_type == 'Put':
                return cdfNormal_dminus*self._strike_price*np.exp(-1*self._riskfree_rate*expiry_time_years) - cdfNormal_dplus*self._spot_price
            else:
                self.logger.debug("Option type not recognized")


def main():
    global logger
    logger = create_logger()
    args = get_cmdline_args()
    option_value= OptionContractPricing(args.spot_price, args.strike_price, args.riskfree_rate, args.expiry_time, args.volatility, args.dividend_yield, args.option_type, logger)
    if args.verbose:
        logger.info(f"{option_value.__str__()}")
        logger.info("\n")
        logger.info(f"{option_value.__repr__()}")
        logger.info(f"Payoff of {args.option_type} option contract using Multi-period Bionomial-Pricing valuation : {option_value.BinomialPricingModel(1.24, 0.80645)}")
        logger.info(f"Payoff of {args.option_type} option contract using Black-Scholes Pricing valuation : {option_value.BlackScholesModel()}")
    elif args.quiet:
        logger.info(f"Payoff of {args.option_type} option contract using Multi-period Bionomial-Pricing valuation : {option_value.BinomialPricingModel(1.24, 0.80645)}")
        logger.info(f"Payoff of {args.option_type} option contract using Black-Scholes Pricing valuation : {option_value.BlackScholesModel()}")
    else:
        pass

if __name__=="__main__":
    main()
        
                
            
        



    
