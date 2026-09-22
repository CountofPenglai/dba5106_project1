"""Compatibility entrypoint for experiments.lasso_training_windows."""
if __name__ == "__main__":
    from experiments.lasso_training_windows import main
    main()
else:
    from portfolio._compat import alias_module
    alias_module(__name__, "experiments.lasso_training_windows")
