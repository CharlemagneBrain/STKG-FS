import os
import pandas as pd
import folium
from folium.plugins import HeatMap, HeatMapWithTime
from folium.plugins import BeautifyIcon
import numpy as np
current_dir = os.path.dirname( __file__ )
'''
colors = ['green', 'red', 'yellow', 'black','yellow', 'cyan', 'orange', 'pink', 'magenta','brown',
          'chartreuse','saddlebrown','teal','violet','greenyellow','tan','darksalmon','darkgoldenrod','mediumblue','aqua',
          'olive','lightgray','palegoldenrod','royalblue','darkturquoise','antiquewhite','darkmagenta'] 
'''
colors = ['blue', 'red', 'darkgreen','purple']
# {'red', 'purple', 'darkgreen', 'black', 'blue', 'orange', 'darkred', 'lightblue', 'lightred', 'darkpurple', 'lightgray', 'darkblue', 'cadetblue', 'lightgreen', 'pink', 'beige', 'green', 'gray', 'white'}.

#==========================
def loadCSVData(_filepath, _separator=","):
    with open(_filepath) as csv_file:
        #df = pd.read_csv(_filepath, delimiter=_separator, header=0,names=['Departement', 'longitude','latitude','type','flood_occurrences'])
        #Place,type,latitude,longitude,Flood,Fire,Drought,Storm,Tornado,FloodPeak,Thunderstorm,Desertification,Deforestation,Landslide,Collapse
        df = pd.read_csv(_filepath, delimiter=_separator, header=0,names=['place','type','latitude','longitude','Flood','Fire','Drought','Storm','Tornado','FloodPeak','Thunderstorm','Desertification','Deforestation','Landslide','Collapse','AgriculturalCampaign','PriceIncrease'])
        return df
#==========================
def spatialViz(_data):
    data = _data
    locations = data[['latitude', 'longitude']]
    locationlist = locations.values.tolist()
    print("data size:",len(locationlist))
    #'Flood','Fire','Drought'
    places = data[['place','type']]
    placeslist=places.values.tolist()
    envrisk = data[['Flood','Fire','AgriculturalCampaign','PriceIncrease']]
    envrisklist=envrisk.values.tolist()
    print(envrisk)
    #heatmap_data = data[['latitude', 'longitude','flood_occurrences']]
   
    #print(heatmap_data)
    
    #map = folium.Map(location=(locationlist[0][0], locationlist[0][1]), zoom_start=6, tiles="cartodb positron", min_zoom = 5, max_zoom = 7)
    map = folium.Map(location=(12.27793,-1.573062), zoom_start=6, tiles="OpenStreetMap")
    '''
    folium.TileLayer('OpenStreetMap').add_to(map)
    folium.TileLayer('Cartodb Positron').add_to(map)
    folium.TileLayer('Cartodb dark_matter').add_to(map)
    folium.TileLayer('OPNVKarte').add_to(map)
    folium.TileLayer('CyclOSM').add_to(map)
    folium.LayerControl().add_to(map)
    '''
    
    #m = folium.Map(location = [51.1657,10.4515],
               #zoom_start=6,
               #min_zoom = 5,
               #max_zoom = 7)
    
    radius = 10
    for i in range(len(data)):
        '''
        folium.Marker(
                location =[locationlist[i][0], locationlist[i][1]],
                icon = folium.Icon(color='green'),
                #popup=data.iloc[i]['name'],
            ).add_to(map)
        '''
        
        for k in range(4) : 
            if envrisklist[i][k]==0:
                continue
            shape=None
            if placeslist[i][1]=="region":
                shape = "circle"
            else :
                if placeslist[i][1]=="departement":
                    shape = "star"
                else:
                    if placeslist[i][1]=="village":
                        shape = "marker"
            folium.Marker([locationlist[i][0], locationlist[i][1]], 
                icon=BeautifyIcon(
                    location=[locationlist[i][0], locationlist[i][1]],
                    #border_color="#00ABDC",
                    #text_color="#00ABDC",
                    border_color=colors[k],
                    text_color=colors[k],
                    number=int(envrisklist[i][k]),
                    inner_icon_style="margin-top:0;font-size:16px",
                    #icon_shape='rectangle-dot', 
                    #icon_shape="marker"
                    #icon_shape="circle"
                    icon_shape=shape,
                    icon_size=(35, 35)
                ),
                tooltip=placeslist[i][0]+":"+"{} events".format(int(envrisklist[i][k]))
                ).add_to(map)


    '''
    HeatMap(heatmap_data, 
        min_opacity=0.4,
        blur = 18
               ).add_to(folium.FeatureGroup(name='Heat Map').add_to(map))
    folium.LayerControl().add_to(map)
    '''


    #map.save(current_dir+"/output_data/dataviz/footprint.html")
    map.save(current_dir+"/output_data/dataviz/carto_all_events.html")

#==========================
if __name__ == '__main__' :
    #dep_risks_file=current_dir+"/input_data/department_risks.csv"
    #dep_risks_file=current_dir+"/input_data/year_flood_by_locations_edited.csv"
    #dep_risks_file=current_dir+"/input_data/all_risks_cities.csv"
    #dep_risks_file=current_dir+"/input_data/env_and_eco_priceandagri.csv"
    dep_risks_file=current_dir+"eco-env-agr.csv"
    data = loadCSVData(dep_risks_file)
    #print(data)
    spatialViz(data)