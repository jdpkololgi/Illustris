"""Separate six-phase authority; old mixture/pilot restrictions are unchanged."""
from datetime import datetime, timezone

PHASES = tuple(f'ph{i:03d}' for i in range(12, 18))
SEEDS = (17, 29)
SCHEMA = 'wholefield-fullcap-generation-v1'


def validate(record):
    expected = dict(schema=SCHEMA, phases=list(PHASES), seeds=list(SEEDS),
                    draws=32, nfe=128, coarse_step=13312, fine_step=26624,
                    method='learned', padding=32, gpu_hour_limit=250,
                    maximum_simultaneous_allocations=2, interactive_only=True,
                    sealed_phases=['ph018', 'ph019'],
                    operator='fullcap_free_space_plus_conservative_linear_density_correction',
                    deadline_utc='2026-10-23T19:17:04+00:00', authorization_commit='35beaf7')
    if any(record.get(k) != v for k, v in expected.items()):
        raise PermissionError('Outside the approved full-cap generation contract')


def guard(record, phase, seed, benchmark=False, freeze=None):
    validate(record)
    if phase not in PHASES or type(seed) is not int or seed not in SEEDS or benchmark or freeze is not None:
        raise PermissionError('Only registered six-phase full-length baseline generation')


def address(phase, cap, seed, first):
    if (phase not in PHASES or cap not in ('NGC', 'SGC') or type(seed) is not int
            or seed not in SEEDS or type(first) is not int or first not in range(0, 32, 4)):
        raise PermissionError('Unregistered phase/cap/model/draw address')
    return f'{phase}/{cap}/seed{seed}/d{first:03d}'


def admit(record, reserved_gpu_hours, gpu_count, wall_seconds, now=None):
    validate(record)
    now = datetime.now(timezone.utc) if now is None else now
    if (type(gpu_count) is not int or gpu_count not in (1, 2, 4)
            or not 0 < wall_seconds <= 4*3600 or reserved_gpu_hours < 0):
        raise PermissionError('Invalid resource request')
    deadline = datetime.fromisoformat(record['deadline_utc'])
    if now.timestamp()+wall_seconds > deadline.timestamp():
        raise PermissionError('Campaign deadline would be exceeded')
    if reserved_gpu_hours+gpu_count*wall_seconds/3600 > record['gpu_hour_limit']:
        raise PermissionError('Allocated GPU-hour cap would be exceeded')

