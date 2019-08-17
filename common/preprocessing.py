from common.utils import Utils

class Preprocessing():
    def __init__(self):
        self.utils = Utils()

    def process(self, img):
        img = self.utils.get_otsu_threshold(img)
        img = self.utils.align_image(img)
        img = self.utils.crop_signature_fast(img)
        img = self.utils.resize(img)
        return img
