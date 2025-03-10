from statsmodels.discrete.discrete_model import (
    Probit, NegativeBinomial, MNLogit, Poisson, CountModel,
    MultinomialModel, GeneralizedPoisson)
from statsmodels.formula.api import logit
from statsmodels.discrete.count_model import (
    ZeroInflatedNegativeBinomialP, ZeroInflatedPoisson)
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.truncated_model import (
    TruncatedLFPoisson, TruncatedLFNegativeBinomialP,
    _RCensoredPoisson, HurdleCountModel)
import pandas as pd

nb_zi = [x.lower() for x in [
    "ZeroInflatedNegativeBinomialP", "ZeroInflatedNegativeBinomialP",
    "nbinom_zinf", "nbinom_zinfl",
    "negativebinomial_zinf", "negativebinomial_zinfl"]]
nb_zi = dict(zip(nb_zi, [ZeroInflatedNegativeBinomialP] * len(nb_zi)))

models = {"orderedmodel": OrderedModel, "ordinal": OrderedModel,
          "negativebinomial": NegativeBinomial, "nbinom": NegativeBinomial,
          **nb_zi, "generalizedpoisson": GeneralizedPoisson,
          "poisson": Poisson, "logit": logit, "binary": logit,
          "probit": Probit, "mnlogit": MNLogit,
          "count": CountModel, "countmodel": CountModel,
          "multinomial": MultinomialModel,
          "multinomialmodel": MultinomialModel,
          "poisson_zinf": ZeroInflatedPoisson,
          "poisson_zinfl": ZeroInflatedPoisson,
          "zip": ZeroInflatedPoisson,
          "ZeroInflatedPoisson".lower(): ZeroInflatedPoisson,
          "hurdle": HurdleCountModel, "hurdlecountmodel": HurdleCountModel,
          "truncatedlfpoisson": TruncatedLFPoisson,
          "truncatedlfnegativebinomialp": TruncatedLFNegativeBinomialP,
          "rcensoredpoisson": _RCensoredPoisson,
          "truncatedpoisson": TruncatedLFPoisson,
          "truncatednegativebinomial": TruncatedLFNegativeBinomialP, }


def print_models():
    """Print the available models."""
    print("Available models:")
    for model in pd.Series(models.keys()).sort_values():
        print(f"\"{model}\": {[models[model]]}")
    return None
