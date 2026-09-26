"""Project-scoped review rounds, distinct from training epochs."""
import random
from compag_annotator.storage.files import uid

def plan(images,count,order='import',seed=42,image_ids=None):
    pool=[i for i in images if i['role']=='pool' and not i.get('removed')]
    n=len(pool)
    if type(count) is not int or not 1<=count<=n:raise ValueError(f'Choose 1 to {n} rounds for {n} eligible pool images')
    ids=[i['id'] for i in pool]
    if order=='shuffle':random.Random(seed).shuffle(ids)
    elif order=='manual':
        if not isinstance(image_ids,list) or len(image_ids)!=n or set(image_ids)!=set(ids):raise ValueError('Manual order must contain each eligible image exactly once')
        ids=list(image_ids)
    elif order!='import':raise ValueError('Unknown round ordering')
    q,r=divmod(n,count);start=0;rounds=[]
    for number in range(count):
        size=q+(number<r)
        rounds.append({'id':uid(),'number':number+1,'image_ids':ids[start:start+size],'status':'planned','model_id':None})
        start+=size
    return {'rounds':rounds,'count':count,'eligible_images':n,'sizes':[len(x['image_ids']) for x in rounds],'order':order,'seed':seed,'image_hashes':{i['id']:i['sha256'] for i in pool}}
