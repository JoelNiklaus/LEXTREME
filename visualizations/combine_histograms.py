import cv2
from pathlib import Path


datasets_with_hierarchical_models = ['brazilian_court_decisions','greek_legal_code','swiss_judgment_prediction','multi_eurlex']

histograms = Path('histograms/')
relevant_images = list()
for image in histograms.glob('**/*'):
    if str(image).endswith('jpg') and 'language' not in str(image):
        for dataset_name in datasets_with_hierarchical_models:
            if dataset_name in str(image):
                relevant_images.append(image)
relevant_images_read = [cv2.imread(str(image)) for image in relevant_images]

# vertically concatenates images
# of same width
im_v = cv2.vconcat(relevant_images_read)

# save output image
cv2.imwrite('histograms/Histograms_for_datasets_with_hierarchical_models.jpg', im_v)
