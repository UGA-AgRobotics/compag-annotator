"""Preserve source bytes; orient a managed derivative, never resize canonical images."""
from pathlib import Path
import io,shutil,warnings
from PIL import Image,ImageOps,UnidentifiedImageError
from compag_annotator.config import MAX_IMAGE_PIXELS,MAX_UPLOAD_BYTES
from .files import digest,uid
FORMATS={'JPEG','MPO','PNG','BMP','WEBP','TIFF'}

def import_media(project_root,path,storage='copy'):
    path=Path(path).expanduser().resolve(strict=True)
    if not path.is_file() or path.stat().st_size>MAX_UPLOAD_BYTES:raise ValueError('Image missing or exceeds 256 MiB limit')
    if storage not in ('copy','reference'):raise ValueError('Choose managed copy or explicit reference')
    with warnings.catch_warnings():
        warnings.simplefilter('error',Image.DecompressionBombWarning)
        with Image.open(path) as im:
            source_format=im.format;source_frames=getattr(im,'n_frames',1)
            if source_format not in FORMATS:raise ValueError(f'Unsupported image format: {source_format}')
            if source_frames!=1 and source_format!='MPO':raise ValueError('Multipage/animated images are not supported in v1')
            # MPO JPEGs can contain an auxiliary picture (for example a gain map).
            # Annotation coordinates always use the full-resolution primary picture.
            if source_format=='MPO':im.seek(0)
            if im.mode in ('I','F','I;16','I;16B','I;16L') or ('bits' in im.info and im.info['bits']>8):raise ValueError('High-bit-depth images require an explicit external conversion')
            if im.width*im.height>MAX_IMAGE_PIXELS:raise ValueError('Image exceeds 100 megapixel safety limit')
            orientation=im.getexif().get(274,1);source_size=[im.width,im.height]
            normalized=ImageOps.exif_transpose(im).convert('RGB');normalized.load()
    identity=digest(path);image_id=uid();root=Path(project_root)
    directory=root/'media'/image_id;directory.mkdir(parents=True)
    original=directory/('original'+path.suffix.lower())
    if storage=='copy':shutil.copy2(path,original);original_ref=str(original.relative_to(root))
    else:original_ref=str(path)
    display=directory/'image.png';normalized.save(display)
    thumb=normalized.copy();thumb.thumbnail((256,256));thumb.save(directory/'thumb.jpg',quality=85)
    record={'id':image_id,'name':path.name,'sha256':identity,'width':normalized.width,'height':normalized.height,'source_size':source_size,'exif_orientation':orientation,'coordinate_contract':'EXIF-oriented original pixels; original bytes preserved','normalized_sha256':digest(display),'original':original_ref,'path':str(display.relative_to(root)),'thumbnail':str((directory/'thumb.jpg').relative_to(root)),'storage':storage,'role':'pool','group_id':identity,'complete':False,'review_actor':None,'revision':0,'removed':False}

    record.update(source_format=source_format,source_frame_count=source_frames,selected_frame=0)
    if source_format=='MPO':record['import_note']='MPO: primary image imported at full resolution; additional embedded images remain in the original file and are not separate annotation images. No HDR gain-map rendering is applied.'
    return record
