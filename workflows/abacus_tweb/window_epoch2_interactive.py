"""Stop the pinned window trainer at its first durable post-epoch-2 checkpoint."""
from pathlib import Path
from workflows.abacus_tweb import p10_train_arm_a as trainer, p12a_encoder_sweep as sweep

def main():
    save = trainer.atomic_torch_save
    def save_then_stop(payload, path):
        save(payload, path)
        if Path(path).name != "arm_a_checkpoint.pt":
            return
        state = payload["resume"]
        if (state["epoch"], state["cursor"]) == (3, 0):
            row = payload["history"][-1]
            if not (row["epoch"] == 2 and row["complete_epoch_coverage"]
                    and row["global_step"] == payload["global_step"] == 692728
                    and set(row["validation"]["per_phase"]) == {"ph006", "ph012", "ph013"}):
                raise RuntimeError("Unexpected validation record")
            print("EPOCH2_VALIDATED_AND_CHECKPOINTED", flush=True)
            raise SystemExit(0)
    trainer.atomic_torch_save = save_then_stop
    sweep.main()

if __name__ == "__main__":
    main()
