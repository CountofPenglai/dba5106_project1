"""Compatibility entrypoint for the Elastic Net portfolio experiment."""
if __name__ != "__main__":
    from portfolio._compat import alias_module
    alias_module(__name__, "experiments.elastic_net_portfolio")
else:
    from experiments.elastic_net_portfolio import main
    main()
