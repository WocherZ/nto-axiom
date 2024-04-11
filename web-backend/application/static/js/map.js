//data_from_django = {{ data_to_map }};
//console.log(data_from_django);

const map = new mapgl.Map('container', {
    center: [37.668598, 55.76259],
    zoom: 13,
    key: '4783e6c0-e59d-4928-91b2-f2f1749b92e4',
});

const directions = new mapgl.Directions(map, {
    // This key can be used for demo purpose only!
    // You can get your own key on http://partner.api.2gis.ru/
    directionsApiKey: '4783e6c0-e59d-4928-91b2-f2f1749b92e4',
});
const markers = [];

let firstPoint;
let secondPoint;
// A current selecting point
let selecting = 'a';
const buttonText = ['Choose two points on the map', 'Reset points'];

const controlsHtml = `<button id="reset" disabled>${buttonText[0]}</button> `;
new mapgl.Control(map, controlsHtml, {
    position: 'topLeft',
});
const resetButton = document.getElementById('reset');

resetButton.addEventListener('click', function() {
    selecting = 'a';
    firstPoint = undefined;
    secondPoint = undefined;
    directions.clear();
    this.disabled = true;
    this.textContent = buttonText[0];
});

map.on('click', (e) => {
    const coords = e.lngLat;

    if (selecting != 'end') {
        // Just to visualize selected points, before the route is done
        markers.push(
            new mapgl.Marker(map, {
                coordinates: coords,
                icon: 'https://docs.2gis.com/img/dotMarker.svg',
            }),
        );
    }

    if (selecting === 'a') {
        firstPoint = coords;
        selecting = 'b';
    } else if (selecting === 'b') {
        secondPoint = coords;
        selecting = 'end';
    }

    // If all points are selected — we can draw the route
    if (firstPoint && secondPoint) {
        directions.pedestrianRoute({
            points: [firstPoint, secondPoint],
        });
        markers.forEach((m) => {
            m.destroy();
        });
        resetButton.disabled = false;
        resetButton.textContent = buttonText[1];
    }
});