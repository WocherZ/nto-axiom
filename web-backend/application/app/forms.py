from django import forms

from .models import City, ImageModel

class ImageForm(forms.ModelForm):
    class Meta:
        model = ImageModel
        fields = ['image']

class CityForm(forms.Form):
    city = forms.ChoiceField(choices=[(city.name, city.name) for city in City.objects.all()])
    image = forms.ImageField(required=False)
    text_description = forms.CharField(widget=forms.Textarea(), required=False)

MOVEMENT_CHOICES = [
    ('pedestrian', 'Пешком'),
    ('jam', 'На авто')
]
class MapForm(forms.Form):
    type_movement = forms.ChoiceField(choices=MOVEMENT_CHOICES)
