from django import forms
from upload_img.models import Image

class ImageForm(forms.ModelForm):
    """Form for the image model"""
    class Meta:
        model = Image
        fields = '__all__'
