import os

from django.db import models
import base64

class ImageModel(models.Model):
    image = models.ImageField(upload_to='static/img/')

    def __str__(self):
        return self.image.url

    def get_base64(self):
        # print(self)
        # print(os.path.abspath('.' + self.image.url))
        with open(os.path.abspath('.' + self.image.url), 'rb') as image_file:
            encoded_string: bytes = base64.b64encode(image_file.read())
            return encoded_string

class City(models.Model):
    name = models.CharField(max_length=100, unique=True)
    latitude = models.FloatField()
    longitude = models.FloatField()
    def __str__(self):
        return self.name



CITIES = [
    {'name': 'Нижний Новгород',
     'latitude': 56.3287,
     'longitude': 44.002
    },
    {'name': 'Ярославль',
     'latitude': 57.6299,
     'longitude': 39.8737
    },
    {'name': 'Владимир',
     'latitude': 56.1366,
     'longitude': 40.3966
    },
    {'name': 'Екатеринбург',
     'latitude': 56.8519,
     'longitude': 60.6122
    },
]
def init_cities(cities):
    # Добавление записей
    for city in cities:
        city = City(name=city['name'],
                    latitude=city['latitude'],
                    longitude=city['longitude']
                    )
        city.save()
