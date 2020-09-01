from IPython import display
import imageio
from glob import glob

# root_read = "E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\AugCompare"
root_read = "E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\CountourCompare"
root_write = "E:\\dataset\\Tongue\\tongue_dataset_tang_plus\\backup\\AugCompare\\GIF"
anim_file = root_write +'/AngleStep1_My1dSortedNpInterpNoBoundaryImages.gif'


with imageio.get_writer(anim_file, mode='I') as writer:
    # path_read =  root_read + '/*.jpg'
    # print("path_read", path_read)
    filenames = glob(root_read + '/*.jpg')
    print(filenames)
    filenames = sorted(filenames)
    print(len(filenames))
    last = -1
    for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
          last = frame
        else:
          continue
        image = imageio.imread(filename)
        writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

# import IPython
# if IPython.version_info >= (6,2,0,''):
#   display.Image(filename=anim_file)