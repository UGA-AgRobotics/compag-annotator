# Training readiness update — 1.0.0rc12

## Training readiness and missing validation images

The Train page reports reviewed training and validation image counts, accepted
training instances by class, excluded images, and steps to fix each blocker.
It rechecks the selected round and polygon-conversion setting before starting;
blocked data stays on Train without creating a failed training job. Refresh
readiness preserves the model, device, epoch, batch and other form settings.
Resuming a saved training job still uses its original frozen snapshot.

The existing execution minimum is one fully reviewed training image with at
least one accepted segmentation instance and one separate fully reviewed
validation image. This minimum does not establish model accuracy. Increasing
epochs or adding masks to a training image does not create validation data.

In Images, change the Dataset role of a separate image to `validation`. Annotate
and review all objects, accept or reject every proposal, then choose **Mark
reviewed** and confirm full image review. Keep at least one other reviewed image
as `pool` or `train`. Related image groups stay in one role. No role, annotation,
review attestation, or dataset split is changed automatically.

This is an incremental usability correction to rc11. Model dependencies and scientific inclusion rules are unchanged. Automated regression tests do not constitute human review or new model-quality evidence.
