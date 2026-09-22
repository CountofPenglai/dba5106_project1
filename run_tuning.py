"""Compatibility entrypoint for experiments.lasso_reward_exposure_search."""
if __name__ == "__main__":
    from experiments.lasso_reward_exposure_search import main
    main()
else:
    from portfolio._compat import alias_module
    alias_module(__name__, "experiments.lasso_reward_exposure_search")
