"""Compatibility entrypoint for the PCA regularized portfolio experiment."""
if __name__ != "__main__":
    from portfolio._compat import alias_module
    alias_module(__name__, "experiments.pca_regularized_portfolio")
else:
    from experiments.pca_regularized_portfolio import main
    main()
