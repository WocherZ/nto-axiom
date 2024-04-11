import time
import os

from django.shortcuts import render, redirect, reverse
import json
import requests

from .models import CITIES, init_cities, ImageModel, City
from .forms import CityForm, ImageForm, MapForm
from .map import generate_map_route, generate_map_points


def home(request):
    #init_cities(CITIES)
    return render(request, 'home.html')

MODEL_SERVICE_URL = 'http://127.0.0.1:8081'

def form(request):
    if request.method == 'POST':
        city_form = CityForm(request.POST, request.FILES)
        if city_form.is_valid():
            selected_city = city_form.cleaned_data['city']
            request.session['selected_city'] = selected_city

            if city_form.cleaned_data.get('image'):
                image = city_form.cleaned_data['image']
                image_obj = ImageModel.objects.create(image=image)
                image_obj.save()

                # data = {"city": selected_city, "image_b64": image_obj.get_base64()}
                # response = requests.post(MODEL_SERVICE_URL + '/upload-image/', data=data)
                # data = response.json()

                data = dict()
                data['places'] = [{'XID': 69,
                         'name': 'Ярославская соборная мечеть',
                         'image': [[255, 255, 255], [255, 255, 255], [255, 255, 255]],
                         'category': 'Мечеть',
                         'city': 0.01,
                         'coords': (39.892933, 57.632572),
                         'prob': 0.86},
                          {'XID': 69,
                           'name': 'Церковь Спаса на городу',
                           'image': [[255, 255, 255], [255, 255, 255], [255, 255, 255]],
                           'category': 'Церковь',
                           'city': 0.01,
                           'coords': (39.895622, 57.622604),
                           'prob': 0.71},
                        {'XID': 69,
                         'name': 'Церковь Рождества Христова',
                         'image': [[255, 255, 255], [255, 255, 255], [255, 255, 255]],
                         'category': 'Церковь',
                         'city': 0.01,
                         'coords': (39.892933, 57.632572),
                         'prob': 0.64},
                          {'XID': 69,
                           'name': 'Российский государственный академический театр драмы имени Фёдора Волкова',
                           'image': [[255, 255, 255], [255, 255, 255], [255, 255, 255]],
                           'category': 'Театр',
                           'city': 0.01,
                           'coords': (39.875633, 57.636059),
                           'prob': 0.53}]

                request.session['user_input_type'] = 'image'
                # request.session['image_path'] = os.path.abspath('.' + image_obj.image.url)
                request.session['image_path'] = image_obj.image.url
                request.session['places'] = data['places']  # Те же данные, что и text, только с image

            elif city_form.cleaned_data.get('text_description'):
                text_description = city_form.cleaned_data['text_description']
                # data = {"city": selected_city, "text": text_description}
                # response = requests.post(MODEL_SERVICE_URL + '/upload-text/', data=data)
                # data = response.json()

                data = [{'XID': 69,
                         'name': 'Ярославская соборная мечеть',
                         'image': [[255, 255, 255], [255, 255, 255], [255, 255, 255]],
                         'category': 'Мечеть',
                         'city': 0.01,
                         'coords': (39.892933, 57.632572),
                         'prob': 0.86},
                          {'XID': 69,
                           'name': 'Церковь Спаса на городу',
                           'image': [[255, 255, 255], [255, 255, 255], [255, 255, 255]],
                           'category': 'Церковь',
                           'city': 0.01,
                           'coords': (39.895622, 57.622604),
                           'prob': 0.71},
                        {'XID': 69,
                         'name': 'Церковь Рождества Христова',
                         'image': [[255, 255, 255], [255, 255, 255], [255, 255, 255]],
                         'category': 'Церковь',
                         'city': 0.01,
                         'coords': (39.892933, 57.632572),
                         'prob': 0.64},
                          {'XID': 69,
                           'name': 'Российский государственный академический театр драмы имени Фёдора Волкова',
                           'image': [[255, 255, 255], [255, 255, 255], [255, 255, 255]],
                           'category': 'Театр',
                           'city': 0.01,
                           'coords': (39.875633, 57.636059),
                           'prob': 0.53}]
                request.session['user_input_type'] = 'text'
                request.session['text_input'] = text_description
                request.session['places'] = data

            else:
                city_form = CityForm()

                context = {
                    'city_from': city_form,
                    'image_path': request.session['image_path'],
                    'error': "Загрузите картинку либо напишите описание места"
                }
                return render(request, 'form.html', context=context)
            return redirect('map')

    city_form = CityForm()

    context = {
        'city_from': city_form,
        'image_path': request.session['image_path']
    }
    return render(request, 'form.html', context=context)


def map(request):
    if not request.session.get('selected_city'):
        redirect('form')
    city = request.session.get('selected_city')
    context = {'city': city}

    if request.method == 'POST': # Построение маршрута
        map_form = MapForm(request.POST)
        if map_form.is_valid():
            # TODO достать точки из сессии
            start_point = "Ярославская соборная мечеть"
            all_points = [
                ({"x": 39.894167, "y": 57.630555}, "Церковь Рождества Христова"),
                ({"x": 39.884796, "y": 57.627155},
                 "Российский государственный академический театр драмы имени Фёдора Волкова"),
                ({"x": 39.875633, "y": 57.636059}, "Ярославская соборная мечеть"),
                ({"x": 39.895622, "y": 57.622604}, "Церковь Спаса на городу"),
            ]
            route_type = map_form.cleaned_data['type_movement']

            context['is_points_map'] = False
            context['map_info'] = generate_map_route(start_point, all_points, route_type)
            context['places_info'] = request.session['places']
            context['image_path'] = request.session['image_path']

        else:
            print("MapForm isn't valid")

    else: # Отображение карты с точками
        map_form = MapForm()

        # TODO Взять координаты из сессии
        all_points = [
            ({"x": 39.894167, "y": 57.630555}, "Церковь Рождества Христова"),
            ({"x": 39.884796, "y": 57.627155},
             "Российский государственный академический театр драмы имени Фёдора Волкова"),
            ({"x": 39.875633, "y": 57.636059}, "Ярославская соборная мечеть"),
            ({"x": 39.895622, "y": 57.622604}, "Церковь Спаса на городу"),
        ]

        context['is_points_map'] = True
        context['map_form'] = map_form
        context['map_info'] = generate_map_points(all_points)
        context['image_path'] = request.session['image_path']

    context['places_info'] = request.session['places']
    return render(request, 'map.html', context=context)
