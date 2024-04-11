# -*- coding: utf-8 -*-
import requests
from itertools import combinations, permutations
import time
import random
import folium
from shapely.geometry import LineString

API_KEY = '1768663f-ebb0-4a99-bd12-ab458c6bc3f9'
# API_KEY = '3b0ecfbd-5858-466c-810c-c138c01b8eaa' - новый
# API_KEY = '3de042a0-485f-4997-b7f3-6ad74aaa7bcd' - новый
# API_KEY = 'bf619093-ce1b-47a1-ae81-0ff40db51a99' - новый
url = 'https://routing.api.2gis.com/carrouting/6.0.0/global?key=' + API_KEY

def calculate_total_distance(route, distances):
    """Calculate the total distance of a route."""
    total_distance = 0
    for i in range(len(route) - 1):
        pair = (route[i], route[i + 1])
        try:
            total_distance += distances[pair]
        except:
            print('No such route')
            total_distance += random.randint(800, 1500)
    return total_distance

def find_fastest_route(distances, start):
    # Create a set of unvisited points excluding the starting point
    unvisited = set(point for pair in distances.keys() for point in pair if point != start)
    current_point = start
    route = [current_point]
    total_distance = 0

    # While there are unvisited points
    while unvisited:
        nearest_point = min(unvisited, key=lambda x: distances.get((current_point, x), float('inf')))
        distance_to_nearest = distances.get((current_point, nearest_point), float('inf'))

        # Update total distance and route
        total_distance += distance_to_nearest
        route.append(nearest_point)

        # Move to the nearest unvisited point
        current_point = nearest_point
        unvisited.remove(nearest_point)

    return route, total_distance


def find_dist(start_point, end_point):
    return ((start_point[0] - end_point[0]) ** 2 + (start_point[1] - end_point[1]) ** 2) ** 0.5


def generate_map_points(points: [[]]) -> folium.Map:
    m = folium.Map(location=(points[0][0]['y'], points[0][0]['x']), zoom_start=13)
    for point in points:
        folium.CircleMarker(location=(point[0]['y'], point[0]['x']), radius=5, color='red', fill=True,
                            fill_color='red').add_to(m)
        folium.Marker(location=(point[0]['y'], point[0]['x']),
                      icon=folium.DivIcon(html=f"""
                          <div style=' 
                              color: blue;
                              border-radius: 5px;
                              padding: 5px;
                              font-family: Arial, sans-serif;
                              font-size: 12px;
                              text-align: center;
                          '>
                              {point[1]}
                          </div>
                      """)).add_to(m)
    m = m._repr_html_()
    return {'map': m}


def generate_map_route(start_point, all_points, route_type='pedestrian'):
    num_points = len(all_points)

    # get distances between all points
    # 15 points - 6 min
    pairs = list(combinations(all_points, 2))
    all_routes = dict()
    all_durations = []
    distances = dict()
    for points in pairs:
        points = list(points)
        points_names = (points[0][1], points[1][1])
        points = [points[0][0], points[1][0]]

        data = {"points": points, "type": route_type}
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()['result']
            best_duration = 99999999
            best_route = None
            for route in result:
                duration = route['total_duration']
                # distance = route['total_distance']
                if duration < best_duration:
                    best_duration = duration
                    best_route = route
            # all_durations.append(best_duration)
            all_routes[points_names] = best_route
            distances[points_names] = best_duration
        else:
            print('Failed to fetch distance:', response.status_code)
        #time.sleep(3.1)

    # distances = {
    #     ("A", "B"): 51783,
    #     ("A", "C"): 17814,
    #     ("A", "D"): 32625,
    #     ("A", "E"): 33843,
    #     ("A", "F"): 31843,
    #     ("B", "C"): 33015,
    #     ("B", "D"): 34470,
    #     ("B", "E"): 61550,
    #     ("B", "F"): 41550,
    #     ("C", "D"): 75142,
    #     ("C", "E"): 39554,
    #     ("C", "F"): 49554,
    #     ("D", "E"): 39758,
    #     ("D", "F"): 37758,
    #     ("E", "F"): 42758
    # }

    distances_2 = dict()
    for pair in distances.items():
        distances_2[(pair[0][1], pair[0][0])] = pair[1]
    all_distances = dict(list(distances.items()) + list(distances_2.items()))

    # for small numbers less than 11. for 11 points - 1 min, 10 points - 3.5s
    # из-за экспоненциальной сложности ограничим кол-во точек для этого алгоритма

    if num_points <= 11:
        all_routes_2 = list(permutations([point[1] for point in all_points]))
        needed_routes = []

        for i in all_routes_2:
            if i[0] == start_point:
                needed_routes.append(i)

        total_distances = {}
        for route in needed_routes:
            total_distances[route] = calculate_total_distance(route, all_distances)

        # Find the route with the minimum total distance
        best_route = min(total_distances, key=total_distances.get)
        best_distance = total_distances[best_route]
    else:
        best_route, best_distance = find_fastest_route(all_distances, start_point)

    print("Best route:", best_route)
    print("Total distance:", best_distance)

    our_route = []
    for i in range(len(best_route) - 1):
        try:
            our_route.append((all_routes[(best_route[i], best_route[i + 1])], (best_route[i], best_route[i + 1])))
        except:
            our_route.append((all_routes[(best_route[i + 1], best_route[i])], (best_route[i + 1], best_route[i])))

    for point in all_points:
        if point[1] == start_point:
            start = [point[0]['y'], point[0]['x']]
            break

    m = folium.Map(location=start, zoom_start=15)
    i = 0
    end_point = start
    for route_idx in range(len(our_route)):
        route = our_route[route_idx][0]
        names = our_route[route_idx][1]
        i += 1
        flag = 0

        maneuvers = route['maneuvers']

        new_mans = []
        for maneuver in maneuvers:
            if 'outcoming_path' in maneuver:
                new_mans.append(maneuver)
        maneuvers = new_mans

        man = maneuvers[0]
        geom = man['outcoming_path']['geometry'][0]
        geometry = LineString(
            [(float(coord.split()[1]), float(coord.split()[0])) for coord in geom['selection'][11:-1].split(', ')])
        dist = find_dist(geometry.coords[0], end_point)

        man = maneuvers[-1]
        geom = man['outcoming_path']['geometry'][-1]
        geometry = LineString(
            [(float(coord.split()[1]), float(coord.split()[0])) for coord in geom['selection'][11:-1].split(', ')])
        dist2 = find_dist(geometry.coords[-1], end_point)

        if dist <= dist2:
            flag = 0
        else:
            flag = 1
            maneuvers = maneuvers[::-1]

        for man_idx in range(len(maneuvers)):
            man = maneuvers[man_idx]
            geometries = man['outcoming_path']['geometry']
            if flag:
                geometries = geometries[::-1]
            for geom_idx in range(len(geometries)):
                geom = geometries[geom_idx]
                geometry = LineString([(float(coord.split()[1]), float(coord.split()[0])) for coord in
                                       geom['selection'][11:-1].split(', ')])
                folium.PolyLine(locations=geometry.coords, color='blue').add_to(m)
                if geom_idx == 0 and man_idx == 0:
                    if flag:
                        start_point_coords = geometry.coords[-1]
                    else:
                        start_point_coords = geometry.coords[0]
                    folium.CircleMarker(location=start_point_coords, radius=5, color='red', fill=True,
                                        fill_color='red').add_to(m)
                    folium.Marker(location=start_point_coords, icon=folium.DivIcon(
                        html=f"""
                                                  <div style=' 
                                                      color: blue;
                                                      border-radius: 5px;
                                                      padding: 5px;
                                                      font-family: Arial, sans-serif;
                                                      font-size: 12px;
                                                      text-align: center;
                                                  '>
                                                      {names[flag]}
                                                  </div>
                                              """)).add_to(m)
                if geom_idx == (len(geometries) - 1) and man_idx == (len(maneuvers) - 1):
                    end_point = geometry.coords[-1]
                    if flag:
                        end_point = geometry.coords[0]
    folium.CircleMarker(location=geometry.coords[-1], radius=5, color='red', fill=True, fill_color='red').add_to(m)
    folium.Marker(location=geometry.coords[-1],
                  icon=folium.DivIcon(html=f"""
                          <div style=' 
                              color: blue;
                              border-radius: 5px;
                              padding: 5px;
                              font-family: Arial, sans-serif;
                              font-size: 12px;
                              text-align: center;
                          '>
                              {names[1 - flag]}
                          </div>
                      """)).add_to(m)

    m = m._repr_html_()
    return {'map': m, 'time': best_distance // 60}

