from itertools import product
from PIL import Image, ImageFont
import numpy as np
from pydantic import BaseModel


def text_(draw, text, x, y, fill='black', outline='white', size=12):
    font = ImageFont.load_default()

    for x_shift, y_shift in product([-1, 1], [-1, 1]):
        draw.text((x + x_shift, y + y_shift), text, font=font, fill=outline)

    draw.text((x, y), text, font=font, fill=fill)


class Example(BaseModel):
    image: Image.Image
    class_name: str

    class Config:
        arbitrary_types_allowed = True
        allow_mutation = False

    def annotated_image(self):
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        text_(draw, self.class_name, 10, 10)
        return image

    @property
    def _repr_png_(self):
        return self.annotated_image()._repr_png_

    def augment(self, augmenter):
        image = Image.fromarray(
            augmenter.augment(image=np.array(self.image))
        )
        return Example(
            image=image,
            class_name=self.class_name
        )