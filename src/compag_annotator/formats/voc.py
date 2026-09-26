"""Pascal VOC boxes: one-based inclusive integer XML coordinates."""
import math
from xml.etree.ElementTree import Element, SubElement, tostring
from defusedxml import ElementTree as safe_xml

from ..geometry import geometry_bbox
from ._common import FormatError, conversion_loss, read_text, child, relative


def export(root, classes, images, annotations, manifest, *, allow_lossy=False, **options):
    by_image = {im["id"]: [] for im in images}
    names = {label["id"]: label["name"] for label in classes}
    if len(set(names.values())) != len(names):
        raise ValueError("VOC requires unique class names or explicit class remapping")
    for annotation in annotations:
        by_image[annotation["image_id"]].append(annotation)
    directory = root / "annotations"
    directory.mkdir()
    for image in images:
        tree = Element("annotation")
        SubElement(tree, "folder").text = "images"
        SubElement(tree, "filename").text = f"{image['id']}.png"
        size = SubElement(tree, "size")
        for key, value in [("width", image["width"]), ("height", image["height"]), ("depth", 3)]:
            SubElement(size, key).text = str(value)
        SubElement(tree, "segmented").text = "0"
        for annotation in by_image[image["id"]]:
            bbox = geometry_bbox(annotation["geometry"], image["width"], image["height"])
            x0, y0, x1, y1 = bbox
            integer_box = [math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)]
            if annotation["geometry"]["type"] != "box" or bbox != integer_box:
                report = conversion_loss(annotation, image, {"type": "box", "xyxy": integer_box}, "VOC represents integer boxes only; segmentation/subpixel coordinates are lost")
                manifest["loss_report"].append(report)
                if not allow_lossy:
                    raise FormatError(f"VOC annotation {annotation['id']} requires explicit allow_lossy", loss_report=[report])
            obj = SubElement(tree, "object")
            SubElement(obj, "name").text = names[annotation["class_id"]]
            flags = annotation.get("source", {}).get("voc", {})
            for flag in ("truncated", "difficult"):
                SubElement(obj, flag).text = "1" if flags.get(flag) else "0"
            bound = SubElement(obj, "bndbox")
            for key, value in zip(("xmin", "ymin", "xmax", "ymax"), [integer_box[0] + 1, integer_box[1] + 1, integer_box[2], integer_box[3]]):
                SubElement(bound, key).text = str(value)
        (directory / f"{image['id']}.xml").write_bytes(tostring(tree, encoding="utf-8", xml_declaration=True))
    manifest["coordinate_convention"] = "VOC one-based inclusive integer boxes; canonical zero-based exclusive xyxy"


def load(path, context):
    root = path if path.is_dir() else path.parent
    files = sorted(root.rglob("*.xml")) if path.is_dir() else [path]
    seen = set()
    for file in files:
        child(root, file.relative_to(root).as_posix())
        tree = safe_xml.fromstring(read_text(file), forbid_dtd=True, forbid_entities=True, forbid_external=True)
        if tree.tag != "annotation":
            raise ValueError("VOC document must have an annotation root")
        name = relative(tree.findtext("filename"))
        width, height = int(tree.findtext("size/width")), int(tree.findtext("size/height"))
        image = context.image(name, width=width, height=height)
        if image["id"] in seen:
            raise ValueError("Multiple VOC files map to the same image")
        seen.add(image["id"])
        for index, obj in enumerate(tree.findall("object")):
            class_id = context.label(obj.findtext("name"))
            values = [int(obj.findtext("bndbox/" + key)) for key in ("xmin", "ymin", "xmax", "ymax")]
            x0, y0, x1, y1 = values
            geometry = {"type": "box", "xyxy": [x0 - 1, y0 - 1, x1, y1]}
            flags = {key: obj.findtext(key, "0") == "1" for key in ("difficult", "truncated")}
            context.add(image, class_id, geometry, external_id=f"{file.name}:{index}", metadata={"voc": flags})
