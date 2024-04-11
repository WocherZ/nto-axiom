from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import base64
import io
from PIL import Image

app = FastAPI()

@app.post('/upload-image/')
async def upload_image(city: str = Form(...), image_b64: str = Form(...)):
    image_data = base64.b64decode(image_b64)
    file_like_object = io.BytesIO(image_data)
    img = Image.open(file_like_object)
    img.save('./decoded_image.png')  # Сохраните изображение в формате PNG

    def model(image):
        # TODO implementation
        return [
            [1, 'Мечеть Иисуса', 'Музей', 0.99],
            [2, 'Мечеть Бога', 'Храм', 0.89]
        ]

    def get_categories(image):
        # TODO implementation
        return [{
                    'category': 'Парк',
                    'prob': 0.99
                },
                {
                    'category': 'Церковь',
                    'prob': 0.97
                }
            ]

    places_data = []
    for result in model(image_data):
        print(result)
        xid = result[0]
        name = result[1]
        category_name = result[2]
        coords = (37.00000, 52.00000)
        prob = result[3]

        places_data.append(
            {
                'XID': xid,
                'name': name,
                'category': category_name,
                'city': city,
                'coords': coords,
                'prob': prob
            }
        )
    data = {
        'places': places_data,
        'place_categories': get_categories(image_data)
    }
    return data


@app.post('/upload-text/')
async def upload_text(city: str = Form(...), text: str = Form(...)):
    # Здесь должна быть логика обработки текста и города
    # Возвращаемый ответ должен содержать N записей с информацией о тексте

    def model(text):
        # TODO implementation
        return [
            [69, 'Парк Коломенское 1', [[255, 255, 255], [255, 255, 255], [255, 255, 255]], 'Парки/усадьбы/дворцы', 'Москва', (39.892933, 57.632572), 0.99],
            [70, 'Парк Коломенское 2', [[255, 255, 255], [255, 255, 255], [255, 255, 255]], 'Парки/усадьбы/дворцы', 'Москва', (39.884796, 57.627155), 0.89],
            [71, 'Парк Коломенское 3', [[255, 255, 255], [255, 255, 255], [255, 255, 255]], 'Парки/усадьбы/дворцы', 'Москва', (39.875633, 57.636059), 0.87],
            [72, 'Парк Коломенское 4', [[255, 255, 255], [255, 255, 255], [255, 255, 255]], 'Парки/усадьбы/дворцы', 'Москва', (39.895622, 57.622604), 0.76]
        ]

    data = []
    for result in model(text):
        data.append(
            {
                'XID': result[0],
                'name': result[1],
                'image': result[2],
                'category': result[3],
                'city': city,
                'coords': result[5],
                'prob': result[6]
            }
        )
    return data
