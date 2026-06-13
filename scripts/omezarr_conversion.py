import numpy as np
from ome_zarr.writer import write_image


def save_to_zarr(            
                arrays, 
                path, 
                **zarr_parameters):

    """ Saves the chunked arrays in an ome-zarr file. 

    Args: 
        Arrays: a list of chunked arrays
        Path: file path of where to save the data

        zarr_parameters: flexible arguments that you will pass in as metadata. If they are the specific 

    Returns: 
        none, saves an ome-zarr object at specified path. 

    Raises
    """
    assert (arrays != None), "arrays are missing"
    assert (path ! = None), "path is missing"

    try:
        zarr_parameters
        print("Metadata detected. \n")
        print(*zarr_parameters)

    except NameError:
        print("No metadata detected for ", path)

    write_image(
        image = arrays,
        group = path,
        *zarr_parameters
    )




# # I will delete this, this is just to take a peek at how stuff works. 
# path = "test_ngff_image.ome.zarr"

# size_xy = 128
# size_z = 10
# rng = np.random.default_rng(0)
# data = rng.poisson(lam=10, size=(size_z, size_xy, size_xy)).astype(np.uint8)


# # write image documentation: 
# def write_image(
#     image: ArrayLike,
#     group: zarr.Group | str,
#     scale_factors: list[int] | tuple[int, ...] | list[dict[str, int]] | None = (
#         2,
#         4,
#         8,
#         16,
#     ),
#     name: str = "image",
#     method: Methods | None = Methods.RESIZE,
#     scaler: Scaler | None = None,
#     fmt: Format | None = None,
#     axes: AxesType = None,
#     coordinate_transformations: list[list[dict[str, Any]]] | None = None,
#     storage_options: JSONDict | list[JSONDict] | None = None,
#     compute: bool = True,
#     scale: dict[str, float] | None = None,
#     axes_units: dict[str, str] | None = None,
#     **metadata: JSONDict,
# ) -> list:



# write_image(
#     data, path, axes="zyx",scale={"z": 1.0, "y": 0.5, "x": 0.5},
#     axes_units={"z": "micrometer", "y": "micrometer", "x": "micrometer"},

#     )