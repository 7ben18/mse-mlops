# Manual Data Flywheel

This runbook covers the manual feedback loop for labeled doctor uploads:
promote accepted labels into the train split, fine-tune, evaluate, publish the
accepted dataset and model artifacts, and exclude a bad promotion batch if it is
rejected later.

The canonical dataset split names stay unchanged:

- `train`: examples used to fit model weights.
- `val`: validation examples used during training and evaluation.
- `test`: final hold-out examples.
- `future`: reviewed production/demo candidates that are not yet in training.

Dataset files and model checkpoints are DVC-managed. Promotion audit manifests
under `reports/feedback/promotions/` are git-managed.

## Promotion Metadata

Each feedback promotion creates one batch ID:

```text
train_YYYYmmddHHMMSS
```

Promoted metadata rows keep the normal `set` value of `train` and receive these
lineage columns:

- `first_train_batch_id`: the first promotion batch that moved the row into training.
- `first_train_at`: the UTC promotion timestamp.
- `training_enabled`: whether the row is eligible for future training runs.
- `promotion_source`: `future_demo` for uploads that came from `future`, or
  `feedback_upload` for new uploaded images.

## Promote And Fine-Tune

Start from the DVC-tracked dataset and model artifacts:

```bash
dvc pull
make ui-up
```

Upload labeled demo data through the UI from `future`. When enough labeled
uploads are ready, promote them:

```bash
make flywheel-status
make flywheel-promote
make flywheel-dvc-add
```

Fine-tune on the promoted training data:

```bash
make train-docker
dvc add models/finetuned/dinov3_ham10000/best_model.pt
```

You can exclude a known bad batch for a single training run without mutating
metadata:

```bash
make train-docker TRAIN_ARGS="--exclude-training-batches train_20260514123045"
```

## Publish Accepted Results

If the promoted data and retrained model are accepted, commit the DVC pointers
and the git-managed promotion manifest:

```bash
git add data.dvc models/finetuned/dinov3_ham10000/best_model.pt.dvc reports/feedback/promotions/*.json
git commit -m "Promote feedback batch and publish retrained model"
dvc push
git push
```

## Reject Or Exclude A Batch

List promoted batches:

```bash
make flywheel-batch-status
```

If a previously accepted batch is rejected, permanently disable it in metadata:

```bash
make flywheel-exclude-batch BATCH_ID=train_20260514123045
make flywheel-dvc-add
make train-docker
dvc add models/finetuned/dinov3_ham10000/best_model.pt
```

If the batch is reinstated later:

```bash
make flywheel-include-batch BATCH_ID=train_20260514123045
make flywheel-dvc-add
```

After retraining and evaluation, publish the updated DVC pointers and model
checkpoint the same way as an accepted promotion.
