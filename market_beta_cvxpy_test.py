"""Compatibility entrypoint for the market-beta constraint experiment."""
if __name__ != "__main__":
    from portfolio._compat import alias_module
    alias_module(__name__, "experiments.market_beta_constraints")
else:
    from experiments.market_beta_constraints import main
    main()
