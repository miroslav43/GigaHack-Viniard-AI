"""Takes 300 random CERTH training photos straight out of the 19 GB images.zip (HTTP range reads) -> certh/{images,labels}."""
import json, random, os, cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor
from remotezip import RemoteZip
os.chdir('/content/bt')
d = json.load(open('certh_ann/annotations/mimc_train_images.json'))
imgs = random.Random(0).sample(d['images'], 300)
anns = {}
for a in d['annotations']:
    anns.setdefault(a['image_id'], []).append(a['bbox'])
os.makedirs('certh/images', exist_ok=True)
os.makedirs('certh/labels', exist_ok=True)
def job(im):
    z = RemoteZip('https://zenodo.org/api/records/10777647/files/images.zip/content')
    a = cv2.imdecode(np.frombuffer(z.read('images/multiple-instance-multiple-class/train_set/' + im['file_name']), np.uint8), cv2.IMREAD_COLOR)
    H, W = a.shape[:2]
    s = 1280 / max(H, W)
    a = cv2.resize(a, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    stem = 'certh_' + im['file_name'][:-4]
    cv2.imwrite(f'certh/images/{stem}.jpg', a, [cv2.IMWRITE_JPEG_QUALITY, 90])
    lines = [f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}" for x, y, w, h in anns.get(im['id'], [])]
    open(f'certh/labels/{stem}.txt', 'w').write('\n'.join(lines))
done = 0
with ThreadPoolExecutor(6) as ex:
    for _ in ex.map(job, imgs):
        done += 1
        if done % 25 == 0:
            print(f'CERTH {done}/300', flush=True)
print('CERTH_OK')
