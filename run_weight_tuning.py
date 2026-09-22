"""Compatibility entrypoint for experiments.lasso_recency_weight_search."""
if __name__ == "__main__":
    from experiments.lasso_recency_weight_search import main
    main()
else:
    from portfolio._compat import alias_module
    alias_module(__name__, "experiments.lasso_recency_weight_search")
