# A script to run Gunpowder pipeline
# 
# Input: zarr file

# Imports
import gunpowder as gp
import matplotlib.pyplot as plt

# Declare array
zarr_path = '/mnt/efs/dl_jrc/student_data/S-MS/raw_data_omezarr/AR163_section1_1x1__XYPos_0.ome.zarr'
raw = gp.ArrayKey('RAW')

# Declare data source
source = gp.ZarrSource(store = zarr_path, datasets = raw)

# Request a batch 
request = gp.BatchRequest()
# request[raw] = gp.Roi() # deprecated?? 




# Pipeline = sequence of nodes:
pipeline = source

# Build the pipeline
with gp.build(pipeline):

  # Request a batch 
  batch = pipeline.request_batch(request)

# show the content of the batch
print(f"batch returned: {batch}")
plt.imshow(batch[raw].data)