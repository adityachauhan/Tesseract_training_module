import csv
import base64
import os
from PIL import Image
from io import BytesIO
output = "pan_new"
count = 0
with open('query_result.csv',mode = 'r') as f:
    data = csv.DictReader(f)
    #print(data[0])
    first_row = next(data)
    for row in data:
        print(str(row))
        b64_string = str(row) + "=" * ((4 - len(str(row)) % 4) % 4)
        fullpath = os.path.join(output, 'pan' + str(count) + '.png')
        with open(fullpath, "wb") as fh:
            fh.write(base64.b64decode(b64_string))
        count = count + 1

