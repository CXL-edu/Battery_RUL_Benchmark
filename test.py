# from sktime.registry import all_estimators
#
# # for forecaster in all_estimators(filter_tags={"scitype:y": ["multivariate", "both"]}):
# for forecaster in all_estimators(filter_tags={"capability:pred_int": True}):
#     print(forecaster)

from sktime.datasets import load_longley

x, y = load_longley()
print(x)
print(y)