"""Compatibility entrypoint for experiments.grid_smoothness."""
if __name__ == "__main__":
    from experiments.grid_smoothness import main
    main()
else:
    from portfolio._compat import alias_module
    alias_module(__name__, "experiments.grid_smoothness")
