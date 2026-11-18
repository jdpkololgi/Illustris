"""Single-phase independent low-z audit entrypoint, no scheduler submission."""
import argparse
from workflows.sbi import e2e_coupled_contract as c
from workflows.sbi import prepare_cfm_lowz_campaign as campaign
from workflows.sbi.e2e_coupled_audit_worker import qualified,qualify_phase


def run(phase):
    c.require_compute();c.phase_guard(phase);campaign.authority()
    if not qualified(phase):qualify_phase(phase)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',required=True)
    run(parser.parse_args().phase)
