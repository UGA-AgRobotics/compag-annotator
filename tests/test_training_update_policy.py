"""Policy boundaries and failure reporting do not require Torch in the core app."""
import pytest
from compag_annotator.providers.training_updates import training_batch_policy, require_training_updates


@pytest.mark.parametrize('images,batch,effective_batch,nbs', [
    (1, 4, 1, 1), (1, 1, 1, 1), (3, 16, 3, 3), (16, 4, 4, 16),
    (64, 4, 4, 64), (65, 4, 4, 64), (1000, 32, 32, 64),
])
def test_batch_policy_matches_dataset_and_preserves_requested_settings(images, batch, effective_batch, nbs):
    requested = {'batch': batch, 'epochs': 10, 'imgsz': 640, 'device': 'cpu', 'seed': 42}
    effective, policy = training_batch_policy(requested, images)
    assert effective == {**requested, 'batch': effective_batch, 'nbs': nbs}
    assert requested['batch'] == batch and 'nbs' not in requested
    assert policy['requested_batch'] == batch and policy['training_images'] == images


def test_resume_policy_retains_original_schedule():
    options = {'batch': 4, 'epochs': 10, 'nbs': 64}
    effective, policy = training_batch_policy(options, 1, resume=True)
    assert effective == options and effective is not options
    assert policy['resume_schedule_preserved']


@pytest.mark.parametrize('saved_batch,saved_nbs', [(1, 1), (4, 64)])
def test_resume_keeps_checkpoint_effective_batch_instead_of_original_ui_request(saved_batch, saved_nbs):
    options = {'batch': 4, 'epochs': 10}
    effective, policy = training_batch_policy(options, 1, resume=True,
        checkpoint_args={'batch': saved_batch, 'nbs': saved_nbs})
    assert effective == {'batch': saved_batch, 'nbs': saved_nbs, 'epochs': 10}
    assert policy['requested_batch'] == 4 and policy['resume_schedule_preserved']
    assert options == {'batch': 4, 'epochs': 10}


@pytest.mark.parametrize('images', [0, -1, True, 1.5])
def test_policy_rejects_invalid_dataset_counts(images):
    with pytest.raises(ValueError, match='at least one image'):
        training_batch_policy({'batch': 4}, images)


@pytest.mark.parametrize('report', [{}, {'passed': False},
    {'passed': True, 'positive_lr_steps': 0, 'learnable_weights_changed': True},
    {'passed': True, 'positive_lr_steps': 1, 'learnable_weights_changed': False}])
def test_no_effective_update_is_not_training_success(report):
    with pytest.raises(RuntimeError, match='No new model was registered'):
        require_training_updates(report)


def test_positive_rate_and_measured_weight_changes_are_required():
    require_training_updates({'passed': True, 'positive_lr_steps': 1, 'learnable_weights_changed': True,
                              'optimizer_weight_change_verified': True})
