"""Deterministic class-aware full-image bbox NMS; no mask unions or raster copies."""
def stitch_predictions(rows, threshold=.5):
    kept=[]; suppressed=[]
    order=sorted(range(len(rows)),key=lambda i:(-float(rows[i].get('score',0)),i))
    for i in order:
        candidate=rows[i]; a=candidate['bbox']; winner=None
        for j in kept:
            other=rows[j]
            if candidate.get('model_class_index')!=other.get('model_class_index'):continue
            b=other['bbox']
            intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            if not intersection:continue
            union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
            if intersection/union>threshold:
                winner=j;break
        if winner is None:kept.append(i)
        else:suppressed.append({'input_index':i,'kept_input_index':winner,
                                'tile_id':candidate.get('source',{}).get('tile_id')})
    return [rows[i] for i in kept],{'method':'class_aware_bbox_nms','iou':threshold,
                                   'input_count':len(rows),'kept_count':len(kept),
                                   'suppressed_count':len(suppressed),'suppressed':suppressed}
