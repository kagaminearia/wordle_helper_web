from django.db import models
import os
from .wrapper_for_web import *


class Image(models.Model):
    title = "wordle"
    image = models.ImageField(upload_to='images')
    time = 0

    def __str__(self):
        return self.title
    
    # get the name of the uploaded file
    def image_name(self):
        return os.path.basename(self.image.name)
    
    def get_read_img(self):
        path = self.image.path
        # path = "/home/st/wordle_img/media/" + self.image.name
        chars = read_img(path)
        gray = chars[0]
        yellow = chars[1]
        green = chars[2]

    def get_results(self):
        path = self.image.path
        ans = get_res(read_img(path))
        self.time = str(ans[1])
        return str(ans[0])

    def time_cons(self):
        path = self.image.path
        ans = get_res(read_img(path))
        return "Executing in " + str(ans[1]) + " seconds"

    class Meta:
        db_table = "myapp_image"
    
