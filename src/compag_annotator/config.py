from pathlib import Path
from platformdirs import user_data_path

def application_data(path=None):
    root=Path(path) if path else user_data_path('compag-annotator',appauthor=False)
    root=root.expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
    return root

BOUNDARY_POLICY={'enabled':False,'allowed_contexts':['training_preparation','automatic_mask_annotation','prediction_boundary_check'],'auto_run_on_inference':False,'auto_run_on_import':False,'auto_run_on_image_open':False,'auto_run_on_manual_edit_or_label':False,'auto_run_on_round_open_or_close':False,'auto_run_on_export':False,'require_explicit_job_opt_in':True,'require_review_before_applying_geometry_changes':True}
DEFAULT_PROCESSING={'mode':'tiled','preset':'512','width':512,'height':512,'overlap':{'mode':'percent','x':25,'y':25}}
MAX_UPLOAD_BYTES=256*1024**2
MAX_IMAGE_PIXELS=100_000_000
