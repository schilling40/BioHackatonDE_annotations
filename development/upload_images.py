import imageio.v3 as imageio
from skimage.measure import label

from biohack_utils.util import omero_credential_parser, connect_to_omero


def upload_3d_images(conn):
    # Get the CochleaNet image and labels.
    image = imageio.imread("/home/anwai/data/M_LR_000167_R_crop_1137-0669-1044.tif")
    labels = imageio.imread("/home/anwai/data/M_LR_000167_R_crop_1137-0669-1044_annotations.tif")

    # Run connected components and reduce precision
    labels = label(labels).astype("uint16")

    def _upload_volume(curr, iname):
        # Upload the image and corresponding labels
        im = conn.createImageFromNumpySeq(
            (plane for plane in curr),
            sizeZ=curr.shape[0],
            sizeC=1,
            sizeT=1,
            imageName=iname,
        )
        print(f"Created image with ID: {im.id}")

    _upload_volume(image, "Cochlea_Neurons")
    _upload_volume(labels, "Neuron_Segmentation")


def upload_2d_images(conn):
    # Get the CovidIF image and labels.
    from torch_em.data.datasets.light_microscopy.covid_if import get_covid_if_paths
    fpath = get_covid_if_paths("/home/anwai/data/covid_if")[0]

    # Open the image
    from elf.io import open_file
    f = open_file(fpath)
    raw_cell = f["raw/serum_IgG/s0"][:]
    raw_nucleus = f["raw/nuclei/s0"][:]
    label_nucleus = f["labels/nuclei/s0"][:]

    def _upload_image(curr, iname):
        images = [curr]
        image = conn.createImageFromNumpySeq(
            (im for im in images),
            imageName=iname,
        )
        print(f"Created image with ID: {image.id}")

    _upload_image(raw_cell, "CovidIF_Cell_Raw")
    _upload_image(raw_nucleus, "CovidIF_Nucleus_Raw")
    _upload_image(label_nucleus, "CovidIF_Nucleus_Labels")


def main():
    parser = omero_credential_parser()
    args = parser.parse_args()

    conn = connect_to_omero(args)

    # upload_3d_images(conn)
    upload_2d_images(conn)

    conn.close()


if __name__ == "__main__":
    main()
