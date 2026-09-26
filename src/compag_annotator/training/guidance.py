"""Explain the existing training requirements without changing dataset membership."""

MINIMUM_MESSAGE = (
    "Minimum to start: 1 fully reviewed training image with at least 1 accepted "
    "segmentation instance, plus 1 different fully reviewed validation image. "
    "Related images must stay in the same dataset role. This is an execution "
    "minimum, not a guarantee of model quality."
)

EXCLUSION_MESSAGES = {
    "explicit_training_exclusion": "Explicitly excluded from training.",
    "all_instances_excluded": "All accepted objects were excluded from training. This image is omitted, not treated as an empty negative image.",
    "role_excluded": "This dataset role is not used for training or validation.",
    "future_round": "Outside the selected cumulative round.",
    "linked_original_missing": "Relink the missing original image before training.",
    "image_missing": "Restore or relink the missing image before training.",
    "image_incomplete": "Review all objects, then use Confirm review in Images or Mark reviewed in Annotate.",
    "awaiting_human_review": "A person must review the entire image and use Mark reviewed.",
}


def explain_readiness(report, document):
    """Add actionable English guidance; keep original reasons for API compatibility."""
    train, val = report['training_images'], report['validation_images']
    fixes = {
        'Create at least one object class': (
            'missing_classes', 'Create at least 1 object class, then assign and accept its masks.'),
        'No complete reviewed training images': (
            'missing_training_images',
            f'No training images are ready ({train}/1 minimum). In Images, use the pool or train '
            'role for at least 1 image. Annotate its objects, accept or reject every proposal, '
            'then use Confirm review in Images or Mark reviewed in Annotate and confirm full image review.'),
        'Supply separately reviewed fixed validation images; training is not reused as validation': (
            'missing_validation_images',
            f'No validation images are ready ({val}/1 minimum). In Images, set the Dataset role '
            'of at least 1 separate image to validation. Annotate and review the entire image, '
            'then use Confirm review in Images or Mark reviewed in Annotate. Keep at least 1 other reviewed image as pool or train. '
            'Increasing epochs or adding more masks to a training image does not satisfy this requirement.'),
        'Related groups span training and validation': (
            'overlapping_image_groups',
            'Training and validation contain related images from the same group. Keep each '
            'group in one dataset role and choose a separate group for validation.'),
        'Training needs at least one reviewed instance, not only negative images': (
            'missing_training_instances',
            'Training needs at least 1 accepted, reviewed segmentation instance in a training '
            'image. Assign an object class, accept the mask, and mark the full image reviewed.'),
        'unassigned_or_archived_class': (
            'invalid_class', 'Assign this object to an active class and review it again.'),
        'annotation_not_human_reviewed': (
            'unreviewed_annotation', 'Review and accept this object before training.'),
        'box_only_not_segmentation_ground_truth': (
            'box_only_annotation',
            'This object has only a bounding box. Create and accept its polygon or mask, '
            'then mark the full image reviewed before segmentation training.'),
    }
    images = {i['id']: i for i in document['images']}
    annotations = {a['id']: a for a in document['annotations']}
    classes = {c['id']: c['name'] for c in document['classes']}
    def describe(item):
        annotation = annotations.get(item.get('annotation_id'))
        if annotation:
            image = images[annotation['image_id']]
            item.update(image_id=image['id'], image_name=image.get('name',image['id']),
                        annotation_revision=annotation['revision'],
                        class_name=classes.get(annotation['class_id'],'Unassigned'))
    for error in report['errors']:
        code, message = fixes.get(error['reason'], ('invalid_annotation_geometry', error['reason']))
        error.update(code=code, message=message)
        describe(error)
    for item in report.get('excluded_annotations', []):
        describe(item)
    for item in report['excluded']:
        image = images[item['image_id']]
        item.update(name=image.get('name', image['id']), role=image['role'],
                    message=EXCLUSION_MESSAGES.get(item['reason'], item['reason']))
    report['minimum_requirements'] = {
        'training_images': 1, 'validation_images': 1, 'training_instances': 1,
        'separate_image_groups': True, 'message': MINIMUM_MESSAGE,
    }
    report['summary'] = f'Ready images: {train} training, {val} validation.'
    report['image_role_counts'] = {
        role: sum(image['role'] == role for image in document['images'])
        for role in ('pool', 'train', 'validation', 'test', 'excluded')
    }
    return report


def readiness_message(report):
    reasons = ' '.join(e.get('message', e['reason']) for e in report['errors'])
    return f"Training data not ready. {report['summary']} {reasons}"
