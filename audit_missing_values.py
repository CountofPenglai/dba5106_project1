"""Compatibility entrypoint for tools.audit_missing_values."""
if __name__ == "__main__":
    from tools.audit_missing_values import main
    main()
else:
    from portfolio._compat import alias_module
    alias_module(__name__, "tools.audit_missing_values")
