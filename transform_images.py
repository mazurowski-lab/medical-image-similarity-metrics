# apply transformation to all images in a folder, and save them in another folder
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm
from argparse import ArgumentParser

transform_kwarg_counts = {
    'gaussian_blur': 2,
    'adjust_sharpness': 3
}

def main(
        input_folder,
):
    transform_names = [
        'gaussian_blur',
        'adjust_sharpness',
    ]
    for transform_name in transform_names:
        print("Applying transform: ", transform_name)
        for kwarg_idx in range(transform_kwarg_counts[transform_name]):
            if transform_name == 'gaussian_blur':
                transform = T.functional.gaussian_blur
                kwargs = {
                    'kernel_size': [5,9][kwarg_idx]
                          }
            elif transform_name == 'adjust_sharpness':
                transform = T.functional.adjust_sharpness
                kwargs = {
                        'sharpness_factor': [0, 0.5, 2][kwarg_idx]
                        }
            else: 
                raise ValueError('Unknown transform_name')

            # loop through all images in the folder
            # output folder also uses kwargs
            output_folder = input_folder + '_' + transform_name + '-'.join([f'{v}' for v in kwargs.values()])

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
    
            for file_name in tqdm(os.listdir(input_folder)):
                # check that image files
                if not file_name.endswith('.png'):
                    continue
                input_path = os.path.join(input_folder, file_name)
                output_path = os.path.join(output_folder, file_name)
                
                # apply transform
                img = Image.open(input_path)
                img = transform(img, **kwargs)
                img.save(output_path)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image_folder', type=str, required=True, help='input image folder')

    args = parser.parse_args()
    main(
        input_folder = args.image_folder,
    )
