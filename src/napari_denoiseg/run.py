import napari

# create Qt GUI context
from src.napari_denoiseg import DenoiSegQWidget, example_magic_widget

with napari.gui_qt():
    # create a Viewer and add an image here
    viewer = napari.Viewer()

    # create a folder for our data
    if not os.path.isdir('./data'):
        os.mkdir('data')

    noise_level == 'n10'
    link = 'https://zenodo.org/record/5156977/files/DSB2018_n10.zip?download=1'

    # check if data has been downloaded already
    zipPath = "data/DSB2018_{}.zip".format(noise_level)
    if not os.path.exists(zipPath):
        # download and unzip data
        data = urllib.request.urlretrieve(link, zipPath)
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall("data")

    # Loading of the training images
    trainval_data = np.load('data/DSB2018_{}/train/train_data.npz'.format(noise_level))
    train_images = trainval_data['X_train'].astype(np.float32)
    train_masks = trainval_data['Y_train']
    val_images = trainval_data['X_val'].astype(np.float32)
    val_masks = trainval_data['Y_val']


    # custom code to add data here
    viewer.window.add_dock_widget(example_magic_widget())
