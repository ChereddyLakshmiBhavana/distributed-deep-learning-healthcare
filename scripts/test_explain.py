import io
from pathlib import Path

import requests
from PIL import Image

backend = 'https://distributed-deep-learning-healthcare.onrender.com'
img_path = Path(r'c:\Users\chitt\OneDrive\Desktop\Distributed-deep learning-smart-health care\data\raw\chest_xray\test\NORMAL\IM-0001-0001.jpeg')
img = Image.open(img_path)
buf = io.BytesIO()
img.save(buf, format='JPEG')
files = {'file': ('IM-0001-0001.jpeg', buf.getvalue(), 'image/jpeg')}
data = {'model': 'k-nearest_neighbors_model'}

response = requests.post(backend + '/predict/explain', files=files, data=data, timeout=120)
print('status', response.status_code)
print(response.text)
