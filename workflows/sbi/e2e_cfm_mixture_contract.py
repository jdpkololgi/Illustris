"""Bounded, truth-independent whole-model mixture and pilot permissions."""
import hashlib

PHASES = ('ph012', 'ph013')
SEEDS = (17, 29)


def pilot_guard(phase, seed):
    if phase not in PHASES or seed not in SEEDS:
        raise PermissionError('Mixture pilot permits ph012/ph013 and seeds17/29 only')


def sampler_seed_guard(seed, phase, benchmark, freeze, comparison=None):
    if seed not in SEEDS:
        raise PermissionError('Unregistered model seed')
    if comparison is not None and comparison.get('schema') == 'wholefield-fullcap-generation-v1':
        from workflows.sbi.e2e_wholefield_fullcap_contract import guard
        guard(comparison, phase, seed, benchmark, freeze)
        return
    if comparison is not None:
        expected = dict(schema='cfm-model-mixture-generation-v1', phases=list(PHASES),
                        seeds=list(SEEDS), nfe=128, draws=32)
        if (any(comparison.get(k) != v for k, v in expected.items())
                or phase not in PHASES or benchmark or freeze is not None):
            raise PermissionError('Outside authorized development comparison')
        return
    if seed != 17 and (phase not in PHASES or not benchmark or freeze is not None):
        raise PermissionError('Seed29 extension is a development-only shortened pilot')


def component_seed(phase, draw):
    """One addressed fair Bernoulli bit for the entire phase/draw, not per core.

    Independent addressed pseudorandom bits; never balance the realized counts.
    The model choice has no observation, target, component output or score input.
    """
    if phase not in PHASES or type(draw) is not int or not 0 <= draw < 32:
        raise PermissionError('Unregistered phase or draw address')
    address = f'cfm-whole-model-mixture-v1:{phase}:{draw}'.encode('ascii')
    return SEEDS[hashlib.sha256(address).digest()[0] & 1]
