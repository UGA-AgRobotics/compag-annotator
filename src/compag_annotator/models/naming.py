"""Dated model labels; checkpoint paths and activation IDs stay stable."""
from datetime import datetime, timezone
import math
from pathlib import Path


def dated_model(record):
    """Name best.pt YOLO records using saved timestamps, never file mtimes.

    Legacy registrations are explicitly distinguished from recorded training
    completion. The label includes the host timezone and a model ID suffix.
    Reading an older registry does not rewrite it or guess a training date.
    """
    row = dict(record)
    filename = row.get('checkpoint_name') or Path(row.get('path') or row.get('name') or 'model.pt').name
    if row.get('provider') != 'yolo' or filename != 'best.pt':
        return row
    value = row.get('trained_at')
    completed = value is not None
    if value is None:
        value = row.get('created_at')
    try:
        if type(value) in (int, float) and math.isfinite(value):
            instant = datetime.fromtimestamp(value, timezone.utc)
        elif isinstance(value, str):
            instant = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if instant.tzinfo is None:
                raise ValueError('Timestamp requires a timezone')
        else:
            raise ValueError('Timestamp unavailable')
        local = instant.astimezone()
        stamp = local.strftime('%Y-%m-%d %H:%M %Z')
        description = ('Training completed: ' if completed else 'Saved / registered: ') + stamp
    except (ValueError, OverflowError, OSError):
        stamp = 'Date unavailable'
        description = 'Saved date unavailable in this older model record'
    base = Path(filename).stem
    base = {'best': 'Best', 'last': 'Last'}.get(base, base)
    suffix = str(row.get('id') or row.get('sha256') or '')[:8]
    row.update(checkpoint_name=filename, name=f'{base} · {stamp}' + (f' · {suffix}' if suffix else ''),
               saved_time_label=description)
    return row
