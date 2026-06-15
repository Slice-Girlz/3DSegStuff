import gunpowder as gp
from skimage import util


class InvertIntensities(gp.BatchFilter):

  def __init__(self, in_array, out_array):
    self.in_array = in_array
    self.out_array = out_array

  def setup(self):

    # tell downstream nodes about the new array
    self.provides(
      self.out_array,
      self.spec[self.in_array].copy())

  def prepare(self, request):

    # to deliver inverted raw data, we need raw data in the same ROI
    deps = gp.BatchRequest()
    deps[self.in_array] = request[self.out_array].copy()

    return deps

  def process(self, batch, request):

    # get the data from in_array and invert it
    data = util.invert(batch[self.in_array].data)

    # create the array spec for the new array
    spec = batch[self.in_array].spec.copy()
    spec.roi = request[self.out_array].roi.copy()

    # create a new batch to hold the new array
    batch = gp.Batch()

    # create a new array
    inverted = gp.Array(data, spec)

    # store it in the batch
    batch[self.out_array] = inverted

    # return the new batch
    return batch