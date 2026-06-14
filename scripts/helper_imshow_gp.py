import matplotlib.pyplot as plt

# helper function to show image(s), channels first # CYX or BCZYX
def imshow(raw, ground_truth=None, prediction=None, z_plane=None, cmap='grey'):
  if len(raw.shape) == 5:  # BCZYX -> BCYX
    assert z_plane is not None, "z_plane must be specified for 5D (BCZYX) input"
    raw = raw[:, :, z_plane]
  if ground_truth is not None and len(ground_truth.shape) == 5:
    ground_truth = ground_truth[:, :, z_plane]
  if prediction is not None and len(prediction.shape) == 5:
    prediction = prediction[:, :, z_plane]
  rows = 1
  if ground_truth is not None:
    rows += 1
  if prediction is not None:
    rows += 1
  cols = raw.shape[0] if len(raw.shape) > 3 else 1
  fig, axes = plt.subplots(rows, cols, figsize=(10, 4), sharex=True, sharey=True, squeeze=False)
  if len(raw.shape) == 3:
    axes[0][0].imshow(raw.transpose(1, 2, 0), cmap=cmap)
  else:
    for i, im in enumerate(raw):
      axes[0][i].imshow(im.transpose(1, 2, 0), cmap=cmap)
  row = 1
  if ground_truth is not None:
    if len(ground_truth.shape) == 3:
      axes[row][0].imshow(ground_truth[0], cmap=cmap)
    else:
      for i, gt in enumerate(ground_truth):
        axes[row][i].imshow(gt[0], cmap=cmap)
    row += 1
  if prediction is not None:
    if len(prediction.shape) == 3:
      axes[row][0].imshow(prediction[0], cmap=cmap)
    else:
      for i, gt in enumerate(prediction):
        axes[row][i].imshow(gt[0], cmap=cmap)
  plt.show()