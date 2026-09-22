"""Compatibility entrypoint for experiments.lasso_forecast_lp."""
if __name__ == "__main__":
    from experiments.lasso_forecast_lp import main
    main()
else:
    from portfolio._compat import alias_module
    alias_module(__name__, "experiments.lasso_forecast_lp")
