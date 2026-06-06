import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.optimize as sco
import yfinance as yf
from scipy.optimize import OptimizeResult


def _portfolio_daily_performance(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate the portfolio's daily returns and portfolio standard deviation.

    :param weights:
    :param mean_returns:
    :param cov_matrix:
    :return: portfolio standard deviation and return
    """
    portfolio_return = float(mean_returns @ weights)
    portfolio_std = float(np.sqrt(float(weights.T @ cov_matrix @ weights)))

    return portfolio_std, portfolio_return


def _portfolio_volatility(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> float:
    """
    Compute the portfolio's daily volatility.

    The volatility is calculated as the standard deviation of portfolio returns
    using the asset covariance matrix and portfolio weights.

    :param weights: Portfolio weights for each asset. The weights should sum to 1.
    :param mean_returns: Expected daily returns of the assets.
    :param cov_matrix: Covariance matrix of the asset returns.
    :return: Daily portfolio volatility (standard deviation of returns).
    """
    return _portfolio_daily_performance(weights, mean_returns, cov_matrix)[0]


def _efficient_return(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    target: float,
) -> scipy.optimize.OptimizeResult:
    """
    Compute the minimum-volatility portfolio that achieves a target return.

    :param mean_returns: Expected returns of the assets.
    :param cov_matrix: Covariance matrix of asset returns.
    :param target: Target portfolio return.
    :return: Optimization result.
    """
    # number of assets in the portfolio
    num_assets = len(mean_returns)

    # additional arguments passed to the objective function
    args = (mean_returns, cov_matrix)

    # helper function to compute the expected portfolio return
    def portfolio_return(weights):
        return _portfolio_daily_performance(weights, mean_returns, cov_matrix)[1]

    # optimization constraints:
    # 1. Portfolio return must equal the target return.
    # 2. Portfolio weights must sum to 1 (fully invested portfolio).
    constraints = (
        {
            "type": "eq",
            "fun": lambda weights: portfolio_return(weights) - target,
        },
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1,
        },
    )

    # long-only portfolio:
    # each weight is constrained to the interval [0, 1]
    bounds = tuple((0, 1) for _ in range(num_assets))

    # solve the constrained optimization problem:
    # minimize portfolio volatility subject to the constraints above
    result = sco.minimize(
        fun=_portfolio_volatility,
        x0=num_assets * [1.0 / num_assets],
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result


def _efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray, returns_range: np.ndarray) -> list[OptimizeResult]:
    """
    Compute portfolios along the effecient frontier.

    :param mean_returns: Expected returns of the assets
    :param cov_matrix: Covariance matrix of asset returns
    :param returns_range: Sequence of target returns for which efficient portfolios are computed.
    :return: Optimization results corresponding to each target return.
    """
    efficients = []

    # Compute the minimum-volatility portfolio for each target return
    for target_return in returns_range:
        efficients.append(_efficient_return(mean_returns=mean_returns, cov_matrix=cov_matrix, target=target_return))
    return efficients


def main_efficient_frontier(company_ticker: dict, start_date: str, end_date: str, num_simulation: int) -> None:
    """Simulate random portfolios and plot the efficient frontier."""
    # Simulate random portfolios
    portfolio_returns = []
    portfolio_volatility = []
    portfolio_weights = []

    #
    prices_df = pd.DataFrame()
    for i, code in enumerate(company_ticker):
        prices_df[i] = yf.download(tickers=[code], start=start_date, end=end_date)["Close"]

    returns_daily = np.log(prices_df / prices_df.shift(1))[1:]
    mean_returns = returns_daily.mean()
    cov_daily = returns_daily.cov()

    num_companies = len(company_ticker)

    for _ in range(num_simulation):
        # generate random portfolio weights
        weights = np.random.random(num_companies)
        weights /= np.sum(weights)

        # compute portfolio statistics
        portfolio_return = round(np.dot(weights, mean_returns), 5)
        portfolio_vol = round(
            np.sqrt(weights.T @ cov_daily @ weights),
            5,
        )

        portfolio_returns.append(portfolio_return)
        portfolio_volatility.append(portfolio_vol)
        portfolio_weights.append(weights)

    portfolio = {
        "Returns": portfolio_returns,
        "Volatility": portfolio_volatility,
    }

    df = pd.DataFrame(portfolio)

    # plot simulated portfolios
    plt.style.use("seaborn-v0_8")
    ax = df.plot.scatter(
        x="Volatility",
        y="Returns",
        figsize=(10, 6),
    )

    # Compute and plot efficient frontier
    target_returns = np.linspace(0.0001, 0.0005, 50)

    efficient_portfolios = _efficient_frontier(
        mean_returns=mean_returns,
        cov_matrix=cov_daily,
        returns_range=target_returns,
    )

    ax.plot(
        [p["fun"] for p in efficient_portfolios],
        target_returns,
        linestyle="-.",
        color="black",
        label="Efficient Frontier",
    )

    ax.set_xlabel("Volatility")
    ax.set_ylabel("Returns")
    ax.legend(labelspacing=0.8)

    plt.show()


if __name__ == "__main__":
    company_ticker = {"066570.KS": "LG electronics", "035420.KS": "NAVER", "015760.KS": "KEPCO", "009070.KS": "KCTC", "000100.KS": "Yuhan"}
    start_date = "2016-01-04"
    end_date = "2022-06-01"
    num_simulation = 50000
    main_efficient_frontier(company_ticker=company_ticker, start_date=start_date, end_date=end_date, num_simulation=num_simulation)
