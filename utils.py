import pandas as pd
import numpy as np
import requests
import json
import pydeck as pdk
import streamlit as st


## Function for getting postal code, geo coordinates of addresses
def find_postal(add):
    '''With the block number and street name, get the full address of the hdb flat,
    including the postal code, geogaphical coordinates (lat/long)'''
    
    # Do not need to change the URL
    # url= "https://developers.onemap.sg/commonapi/search?returnGeom=Y&getAddrDetails=Y&pageNum=1&searchVal="+ add  
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal=${add}&returnGeom=Y&getAddrDetails=Y&pageNum=1"   
    response = requests.get(url)
    try:
        data = json.loads(response.text) 
    except ValueError:
        print('JSONDecodeError')
        pass
    
    return data


def find_nearest(house, amenity, radius=2):
    
    from geopy.distance import geodesic
    #--edit
    results = pd.DataFrame(data=None)
    stns = []
    ndist = []
    plot_map = {}
    # first column must be address
    for index,flat in enumerate(house.iloc[:,0]):
        
        # 2nd column must be latitude, 3rd column must be longitude
        flat_loc = (house.iloc[index,1],house.iloc[index,2])
        flat_amenity = ['','',100,0]
        amenity_2km = pd.DataFrame({'lat':[], 'lon':[]})

        for ind, eachloc in enumerate(amenity.iloc[:,0]):
            amenity_loc = (amenity.iloc[ind,1], amenity.iloc[ind,2])
            distance = geodesic(flat_loc, amenity_loc)
            distance = float(str(distance)[:-3])
            stns.append(eachloc)  #--edit
            ndist.append(distance) 
            
            if distance <= radius: # compute number of amenities in 2km radius
                flat_amenity[3] += 1
                amenity_2km = amenity_2km.append(pd.DataFrame({'name':[eachloc], 
                                                               'lat':[amenity_loc[0]], 
                                                               'lon':[amenity_loc[1]]}))
            if distance < flat_amenity[2]: # find nearest amenity
                flat_amenity[0] = flat
                flat_amenity[1] = eachloc
                flat_amenity[2] = distance
        
        plot_map[flat]= flat_amenity
    #--edit
    results['amenity']=stns
    results['ndist']=ndist
    results = results.sort_values('ndist')
    return results.iloc[[0]], plot_map, amenity_2km



def dist_from_location(house, location):
    """
    this function finds the distance of a location from the 1st address
    First is a dataframe with a specific format:
        1st column: any string column ie addresses taken from the "find_postal_address.py"
        2nd column: latitude (float)
        3rd column: longitude (float)
    Column name doesn't matter.
    Second is tuple with latitude and longitude of location
    """
    from geopy.distance import geodesic
    results = {}
    # first column must be address
    for index,flat in enumerate(house.iloc[:,0]):
        
        # 2nd column must be latitude, 3rd column must be longitude
        flat_loc = (house.iloc[index,1],house.iloc[index,2])
        distance = geodesic(flat_loc,location)
        distance = float(str(distance)[:3]) # convert to float
    return distance

    
def _max_width_(prcnt_width):
    import streamlit as st
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
    
    
def draw_map(data, lat, lon, zoom, amenities_toggle):
    
    if amenities_toggle[0]:       
        mrts = data[data['type']=='MRT'].drop(['selected_flat'],axis=1)
    else: 
        mrts = None
    if amenities_toggle[1]: 
        schools = data[data['type']=='School'].drop(['selected_flat'],axis=1)
    else: 
        schools = None
    if amenities_toggle[2]: 
        fdctrs = data[data['type']=='Food Centres'].drop(['selected_flat'],axis=1)
    else: 
        fdctrs = None
    if amenities_toggle[3]: 
        cctrs = data[data['type']=='Childcare'].drop(['selected_flat'],axis=1)
    else: 
        cctrs = None
    if amenities_toggle[4]: 
        hdb = None
    else: 
        hdb = data[data['type']=='HDB']

    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v8",
        initial_view_state={
            "latitude": lat,
            "longitude": lon,
            "zoom": zoom,
            "pitch": 50,
        },
        tooltip={"html": "{name}",
                 "style": {"background": "grey", 
                           "color": "white", 
                           "font-family": '"Helvetica Neue", Arial', 
                           "z-index": "10000"}},        
        layers=[
            pdk.Layer( # mrt - yellow
                'ScatterplotLayer',
                data=mrts,
                get_position='[LONGITUDE, LATITUDE]',
                # get_color='[255, 0, 0, 160]',
                get_fill_color='[255, 166, 0, 160]',  #yellow
                pickable=True,
                get_radius=90,
            ),
            pdk.Layer( # schools - green
                'ScatterplotLayer',
                data=schools,
                get_position='[LONGITUDE, LATITUDE]',
                pickable=True,
                # get_color='[0, 102, 255, 160]',
                get_fill_color='[114, 176, 67, 160]',  #green
                get_radius=70,
            ),
            pdk.Layer( # hawkers - violet
                'ScatterplotLayer',
                data=fdctrs,
                get_position='[LONGITUDE, LATITUDE]',
                # get_color='[204, 0, 204, 160]',
                # get_fill_color = '[88, 80, 141, 160]',  #purple
                get_fill_color='[188, 80, 144, 160]',  #pink
                pickable=True,
                get_radius=50,
            ),
            pdk.Layer( # ccentre - blue
                'ScatterplotLayer',
                data=cctrs,
                get_position='[LONGITUDE, LATITUDE]',
                # get_color='[0, 102, 255, 160]',  #blue
                get_fill_color='[67, 176, 241, 160]', 
                pickable=True,
                get_radius=60,
            ),
            pdk.Layer( # HDB user - dark blue
                'ScatterplotLayer',
                stroked=True,
                data=data[data['selected_flat']==1],
                get_position=["LONGITUDE", "LATITUDE"],
                # get_color='[188, 80, 144, 160]',  # green 
                get_fill_color='[0, 63, 92, 160]',  # blue 
                line_width_min_pixels=5,
                # get_line_color=[180, 80, 144, 160],
                get_line_color=[0, 63, 92, 100],
                get_radius=30,
                
            ),
            pdk.Layer( # flats
                "ColumnLayer",
                data=hdb,
                get_position=["LONGITUDE", "LATITUDE"],
                get_elevation=["selected_flat * 1000"],
                #elevation_scale=4,
                elevation_range=[0, 3000],
                radius=40,
                # get_fill_color=["0", "153-(153*selected_flat)", "0", 80],
                get_fill_color=["0", "63-(63*selected_flat)", "92", 80],
                pickable=True,
                auto_highlight=True,
                #extruded=True,
         ),
        ]
    ))      