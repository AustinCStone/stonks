"""Calculates the optimal expercise price to buy given an expected price movement and a certain
number of days until expiration."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_float('current_price', 280., 'The current price of the underlying.', lower_bound=0)
flags.DEFINE_float('risk_free_interest_rate', .0063, 'The current risk free interest rate',
                   lower_bound=0)
flags.DEFINE_float('days_to_expiration', 180, 'The number of days until targeted expiration date.',
                   lower_bound=1)
flags.DEFINE_float('volatility', .4, 'The volatility of the underlying over the next 12 months. '
                   'You can use the implied volatility based on the current pricing or your '
                   'own estimate. One good estimate can be the average of the implied volatility '
                   'across several different options currently on the market. This is in the same '
                   'units robinhood gives when looking at option implied volatility.')
flags.DEFINE_float('expected_price_movement', .3, 'The percent price movement that you expect to'
                   'occur before the expiration date.')
flags.DEFINE_bool('plot', True, 'Whether or not to plot the expected returns for various strike '
                  'prices.')


def plot(exercise_prices, profits, title):
    plt.plot(exercise_prices, profits)
    plt.xlabel('Exercise Price')
    plt.ylabel('Profit (%)')
    plt.title(title)
    plt.show()


def black_scholes(current_price, exercise_price, risk_free_interest_rate,
                  days_to_expiration, volatility, call=True):
    """The black scholes pricing formula."""
    s = current_price
    x = exercise_price
    r = risk_free_interest_rate
    # It is assumed volatility is in units of years. This is the unit robinhood uses for implied
    # volatility. We therefore put time in units of years. Volatility is the standard deviation of
    # log returns over the course of a year.
    t = days_to_expiration / 365.
    v = volatility
    d1 = (np.log(s / x) + (r + (v ** 2.) / 2.) * t) / (v * np.sqrt(t))
    d2 = (np.log(s / x) + (r - (v ** 2.) / 2.) * t) / (v * np.sqrt(t))
    if call:
        return s * norm.cdf(d1) - x * np.exp(-r * t) * norm.cdf(d2)
    else:
        return x * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)


def main(unused):
    min_exercise_price = FLAGS.current_price * .1
    max_exercise_price = FLAGS.current_price * 3.
    print(f'Searching between excercise prices {min_exercise_price} and {max_exercise_price}...')
    max_profit = -np.inf
    best_exercise_price = None
    call = FLAGS.expected_price_movement > 0.
    all_profits = []
    all_exercise_prices = []
    for exercise_price in np.linspace(min_exercise_price, max_exercise_price, 1000):
        current_option_price = black_scholes(FLAGS.current_price, exercise_price,
                                             FLAGS.risk_free_interest_rate,
                                             FLAGS.days_to_expiration,
                                             FLAGS.volatility,
                                             call=call)
        future_underlying_price = FLAGS.current_price * (1. + FLAGS.expected_price_movement)
        if call:
            profit = (future_underlying_price - exercise_price) - current_option_price
            profit /= current_option_price 
        else:  # put option
            profit = (exercise_price - future_underlying_price) - current_option_price
            profit /= current_option_price
        if profit > max_profit:
            max_profit = profit
            best_exercise_price = exercise_price

        if profit > -1.:
            all_exercise_prices.append(exercise_price)
            all_profits.append(profit * 100.)

    print(f'Best exercise price is {best_exercise_price:.2f}, profit is {max_profit * 100:.2f}%.')
    if FLAGS.plot:
        title = f'S: {FLAGS.current_price}, predicted movement: {FLAGS.expected_price_movement}' + \
            f', time: {FLAGS.days_to_expiration}, v: {FLAGS.volatility}'
        plot(all_exercise_prices, all_profits, title)


if __name__ == '__main__':
  app.run(main)