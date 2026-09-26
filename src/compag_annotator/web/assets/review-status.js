// Class assignment and a final review decision are different saved states.
export const needsReview = (object) => ['draft', 'proposal'].includes(object.status);
export function reviewStatus(object) {
  const training = object.training_excluded ? ' · Excluded from training' : '';
  if (object.status === 'accepted')
    return (object.review_actor === 'automated_qa' ? 'Accepted · automated QA' : 'Accepted') + training;
  if (object.status === 'rejected') return 'Rejected' + training;
  if (object.status === 'superseded') return 'Replaced by merge' + training;
  return (object.class_id ? 'Labeled · needs review' : 'Needs label & review') + training;
}
export function reviewCounts(objects) {
  return {
    pending: objects.filter(needsReview).length,
    accepted: objects.filter(o => o.status === 'accepted').length,
    rejected: objects.filter(o => o.status === 'rejected').length,
  };
}
