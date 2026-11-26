"""Retrospective attribution of the return gap versus equal weight."""
import numpy as np

POSITION_TOL = 1e-7


def attribute_returns(weights, returns, winners):
    """Two additive views: actual short/long P&L and asset contributions versus EW."""
    ew = 1 / len(returns.columns)
    shorts = weights.clip(upper=0) * returns
    longs = weights.clip(lower=0) * returns
    gross = (weights * returns).sum(axis=1)
    components = {
        'Losses on rising shorts': shorts.clip(upper=0).sum(axis=1),
        'Gains on falling shorts': shorts.clip(lower=0).sum(axis=1),
        'Long holdings': longs.sum(axis=1),
    }
    np.testing.assert_allclose(sum(components.values()), gross, atol=1e-12, rtol=0)
    components['Portfolio daily return'] = gross

    gap = (weights - ew) * returns
    top = returns.columns.isin(winners)
    short = weights < -POSITION_TOL
    underweight = ~short & (weights < ew - POSITION_TOL)
    groups = []
    for label, columns in [('Top 20', top), ('Other 78', ~top)]:
        group = gap.loc[:, columns]
        parts = []
        for name, positions in [('short', short), ('nonshort below EW', underweight),
                                ('at or above EW', ~(short | underweight))]:
            pnl = group.where(positions.loc[:, columns], 0).sum(axis=1)
            components[f'{label}: {name}'] = pnl
            parts.append(pnl)
        total = group.sum(axis=1)
        np.testing.assert_allclose(sum(parts), total, atol=1e-12, rtol=0)
        components[f'{label}: total gap to EW'] = total
        groups.append(total)
    components['Total gap to EW'] = gross - returns.mean(axis=1)
    np.testing.assert_allclose(sum(groups), components['Total gap to EW'], atol=1e-12, rtol=0)
    return {name: daily.mean() for name, daily in components.items()}
