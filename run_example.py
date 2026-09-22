"""Compatibility entrypoint for experiments.regularized_baselines."""
if __name__ == "__main__":
    from experiments.regularized_baselines import main
    main()
else:
    from portfolio._compat import alias_module
    alias_module(__name__, "experiments.regularized_baselines")
