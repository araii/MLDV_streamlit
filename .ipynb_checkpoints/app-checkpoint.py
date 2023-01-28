import streamlit as st
import numpy as np
import pandas as pd
# from reference
from utils import find_postal, find_nearest, dist_from_location, _max_width_, draw_map, find_nearest_test
import pydeck as pdk
import joblib


## CONSTANTS
feature_cols = ['ndist_fd_centre', 'log_area_sqm', 'ndist_city', 'logndist_school', 'logndist_mrt', 'storey_range', 'flat_type', 'lease_commence_date','flat_model_Adjoined', 'flat_model_Common','flat_model_DoubleStorey', 'flat_model_Premium', 'region_Central', 'region_East', 'region_North','region_North_East', 'region_West']

ref_flatmodel = ['flat_model_Adjoined', 'flat_model_Common', 'flat_model_DoubleStorey', 'flat_model_Premium']

ref_town = ['region_Central','region_East','region_North','region_North_East','region_West']

townlist = ['-Select Town-', 'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH','CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST','KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES','TOA PAYOH', 'WOODLANDS', 'YISHUN']

fmodellist = ['Model A','Improved','New Generation','Standard', 'Simplified','Apartment','Model A2','Multi Generation', '3Gen','2-room','DBSS','Type S1','Type S2', 'Premium Apartment','Terrace','Premium Apartment Loft', 'Improved-Maisonette','Premium Maisonette','Maisonette', 'Model A-Maisonette','Adjoined flat']

ftypelist = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE','MULTI-GENERATION']

storeylist = ['01 TO 03','04 TO 06','07 TO 09','10 TO 12', '13 TO 15','16 TO 18', '19 TO 21','22 TO 24','25 TO 27', '28 TO 30','31 TO 33','34 TO 36', '37 TO 39','40 TO 42','43 TO 45', '46 TO 48','49 TO 51']

#==========================================================================================
st.set_page_config(page_title="HDB Resale Price", layout="wide")
#==========================================================================================

# FUNCTIONS

# load data
@st.cache
def load_data(filepath):
	return pd.read_csv(filepath)

# Map and generate list based on one-hot feature
def matchList(e, key):
	if key==e:
		return 1
	else:
		return 0

# Search word return index
def searchInList(word, lst):
    for i, s in enumerate(lst):
        if word in s:
            return i
    return 0

# Flat coordinates
def getFlatLatLong(flat_address):
	loc = find_postal(flat_address)
	try:
		coord = loc['results'][0]
		return pd.DataFrame({'BLK_NO':[coord['BLK_NO']],
                             'LATITUDE':[coord['LATITUDE']],
							 'LONGITUDE':[coord['LONGITUDE']],
                             'ROAD_NAME':[coord['ROAD_NAME']],
							 'POSTAL':[coord['POSTAL']]})
	except IndexError:
		return st.error('Oops! Address is not valid. Please enter valid address.')

# User input
def getAddrInfo(input_addr):
    flat_coord = getFlatLatLong(input_addr)
    # ---- can't seem to use my distance calculation method -----
    # getNearestDist(flat_coord['LATITUDE'], flat_coord['LONGITUDE'], mrt_coord)
    nearest_mrt = find_nearest(flat_coord, mrt_coord)
    nearest_sch = find_nearest(flat_coord, school_coord)
    nearest_fdctr = find_nearest(flat_coord, fdcentre_coord)
    city = mrt_coord[mrt_coord['STN_NAME']=='ORCHARD MRT STATION']
    city_loc = (float(city['Latitude']), float(city['Longitude']))
    dist_city = dist_from_location(flat_coord, city_loc)
    return flat_coord, nearest_mrt, nearest_sch, nearest_fdctr, dist_city

# User Input
def getEncodedRef():
    # encoded vars
    cc_ftype = tbl_flattype[tbl_flattype['flat_type']==flat_type].iloc[0]['ref']
    cc_storey = tbl_storey[tbl_storey['storey_range']==storey].iloc[0]['ref']
    # one-hot vars
    ht_town = tbl_town[tbl_town['town']==town].iloc[0]['region']
    ht_fmodel = tbl_flatmodel[tbl_flatmodel['flat_model']==flat_model].iloc[0]['ref']
    cc_town = list(map(lambda e: matchList(e, ht_town), ref_town))
    cc_fmodel = list(map(lambda e: matchList(e, ht_fmodel), ref_flatmodel))
    return cc_ftype, cc_storey, ht_town, ht_fmodel, cc_town, cc_fmodel


def getMapInfo(flat_coord):
    m0, m1, mrts_2km = find_nearest_test(flat_coord, mrt_coord)
    mrts = pd.DataFrame.from_dict(m1).T
    mrts = mrts.rename(columns={0: 'flat',
                                      1: 'mrt',
                                      2: 'mrt_dist',
                                      3: 'num_mrt_2km'}).reset_index().drop('index',
                                                                            axis=1)
    mrts_2km['type'] = ['MRT']*len(mrts_2km)
    #
    s0, s1, schs_2km = find_nearest_test(flat_coord, school_coord)
    schools = pd.DataFrame.from_dict(s1).T
    schools = schools.rename(columns={0: 'flat',
                                      1: 'school',
                                      2: 'school_dist',
                                      3: 'num_school_2km'}).reset_index().drop('index', 
                                                                               axis=1)
    schs_2km['type'] = ['School']*len(schs_2km)
    #
    f0, f1, fdctrs_2km = find_nearest_test(flat_coord, fdcentre_coord)
    fdctrs = pd.DataFrame.from_dict(f1).T
    fdctrs = fdctrs.rename(columns={0: 'flat',
                                    1: 'fdctr',
                                    2: 'fdctr_dist',
                                    3: 'num_fdctr_2km'}).reset_index().drop('index', 
                                                                             axis=1)
    fdctrs_2km['type'] = ['Food Centres']*len(fdctrs_2km)
    # concat 
    map_2km = pd.concat([mrts_2km, schs_2km, fdctrs_2km])
    map_2km = map_2km.rename(columns={'lat':'LATITUDE', 'lon':'LONGITUDE'})
    # map_coord
    map_coord = pd.concat([flat_coord, 
                           mrts.drop(['flat'], axis=1),
                           schools.drop(['flat'], axis=1),
                           fdctrs.drop(['flat'], axis=1)],
                         axis=1)
    map_coord['address'] = map_coord['BLK_NO'] +" "+map_coord['ROAD_NAME']
    map_coord['selected_flat'] = [1]
    # flats
    flats = load_data('_datasets/hdb2/HDBResalePrices2012-2023Jan.csv')[['LATITUDE',
                                                                         'LONGITUDE',
                                                                         'block',
                                                                         'street_name',
                                                                         'POSTAL',
                                                                         'year']]
    flats['address']=flats['block']+" "+flats['street_name']+" "+flats['POSTAL'].map(lambda x: str(x))
    flats = flats.drop_duplicates()
    flats = flats[flats['year']>=2020]  # take only from 2020 onwards
    flats = flats.drop(['block','year','POSTAL','street_name'], axis=1)
    flats['selected_flat'] = [0.000001]*len(flats)
    flats = flats.append(map_coord[['LATITUDE', 
                                    'LONGITUDE', 
                                    'selected_flat', 
                                    'address']], ignore_index=True)
    flats[['LATITUDE','LONGITUDE','selected_flat']] = flats[['LATITUDE',
                                                             'LONGITUDE',
                                                             'selected_flat'
                                                            ]].astype(float)
    flats['type'] = ['HDB']*len(flats)
    flats = flats.rename(columns={'address':'name'})
    all_bldgs = pd.concat([map_2km, flats]).reset_index()
    #
    cb1, cb2, cb3, cb4 = st.columns(4)
    show_mrt = cb1.checkbox('MRT Stations',True)
    show_schools = cb2.checkbox('Schools',True)
    show_fdctrs = cb3.checkbox('Food Centres',True)
    hide_hdb = cb4.checkbox('Hide HDBs',False)    
    #
    amenities_toggle= [show_mrt, show_schools, show_fdctrs, hide_hdb]
    draw_map(all_bldgs, 
             float(map_coord.iloc[0]['LATITUDE']), 
             float(map_coord.iloc[0]['LONGITUDE']),
             13.5,
             amenities_toggle)
    return all_bldgs



# LOAD AMENITIES COORDINATES
fdcentre_coord = load_data('_datasets/latlong_fd_centres.csv')[['name_of_centre','Latitude','Longitude']]
mrt_coord = load_data('_datasets/latlong_mrt_stns.csv')[['STN_NAME','Latitude','Longitude']]
school_coord = load_data('_datasets/latlong_schools.csv')[['school_name','Latitude','Longitude']]

# LOAD LOOKUP TABLES
tbl_flatmodel = load_data('_datasets/lookup_flatmodel.csv')
tbl_flattype =  load_data('_datasets/lookup_flattype.csv')
tbl_storey =load_data('_datasets/lookup_storeyrange.csv')
tbl_town = load_data('_datasets/lookup_town_region.csv')
tbl_avgarea = load_data('_datasets/avg_floorarea.csv') # use avg on default
tbl_avgdist_city = load_data('_datasets/avg_dist_city.csv')
tbl_avgdist_fdctr = load_data('_datasets/avg_dist_fd_ctr.csv')
tbl_avgdist_mrt = load_data('_datasets/avg_dist_mrt.csv')
tbl_avgdist_sch = load_data('_datasets/avg_dist_sch.csv')

# LOAD MODEL
model = joblib.load('hdb_predict_logresale_price.pkl')
    


#==========================================================================================
st.title('HDB Resale Price Prediction')

# st.sidebar.write(
#     f"This app shows how a Streamlit app can interact easily with a to read or store data."
# )

# st.write("""
# # Title
# This is description about the app predicts the **blabla** type!
# """)

_max_width_()

c1, c2 = st.columns(2)
c1.write("""**Enter address or postal code**""")
addrType = c2.radio('option:',('Address', 'Postal Code'), horizontal=True)

    
# FORM LAYOUT       
if addrType == "Address":
    st.subheader('Input Blk and Address')
    a, b = st.columns(2)   
    blk = a.text_input('Enter Blk','988B')  # 651A
    addr = b.text_input('Enter Address', 'BUANGKOK GREEN') # Ang Mo Kio
    flat_address = blk+" "+addr
else:
    st.subheader('Input Postal Code') 
    flat_address = st.text_input("Enter Postal Code", '521497')
        

[flat_coord, nearest_mrt, nearest_sch,
nearest_fdctr, dist_city] = getAddrInfo(flat_address)

all_coords = getMapInfo(flat_coord)
# st.write(all_coords)

# search town index from roadname
road = flat_coord.iloc[0]['ROAD_NAME'].split(' ')[0]  
townidx = searchInList(road, townlist) # st.write(road+" : "+str(townidx))

a, b = st.columns(2) 
town = a.selectbox('Town', townlist, index=townidx)
flat_model = b.selectbox('Flat Model', fmodellist, index=0)

flat_type = a.selectbox('Flat Type', ftypelist, index=0, key='sel_flat_type') 
# get avg. floor area
avgarea = int(tbl_avgarea[tbl_avgarea['flat_type']==flat_type].iloc[0]['floor_area_sqm'])
floor_area = b.slider("Floor Area (sqm)", 30,300,avgarea, key='slider',
                     help="""
                     
                     Default value is based on the flat_type.
                     
                     Please adjust the value if you know
                     the floor area of the apartment.
                     
                     """,) 
storey = a.selectbox('Storey', list(storeylist), index=0)
lease_commence_date = b.selectbox('Lease Commencement Date', list(reversed(range(1966, 2019))), index=1)
onPress = st.button(label='✨ GIVE ME AN ESTIMATE!')


    
# PRESS BUTTON
if onPress:
    try:
        cc_ftype, cc_storey, ht_town, ht_fmodel, cc_town, cc_fmodel = getEncodedRef()
        # DISPLAY RESULTS
        expander = st.expander("User Input Parameters")
        with expander:
            # st.subheader('User Input parameters')
            st.write("""**TOWN**: """, town)
            st.write("""**FLAT MODEL**: """, flat_model)
            st.write("""**FLAT TYPE**: """, flat_type)
            st.write("""**FLOOR AREA**: """, str(floor_area))
            st.write("""**STOREY RANGE**: """, storey)
            st.write("""**LEASE COMMENCE YEAR**: """, str(lease_commence_date))
            st.subheader('Class labels and their corresponding index number')

            st.write("""**FLAT LOCATION**: """, flat_coord)
            st.write("""**FLAT LOCATION**: """, flat_coord.iloc[0, 3])
            st.write("""**NEAREST MRT**: """, nearest_mrt)
            st.write("""**NEAREST SCHOOL**: """, nearest_sch)
            st.write("""**NEAREST FOOD CENTER**: """, nearest_fdctr)
            st.write("""**DISTANCE TO CITY**: """, str(dist_city)+"km")
            # Hide this
            
            st.write("""flatype code: """, cc_ftype)
            st.write("""storey code: """, cc_storey)
            # town-region decoding
            st.write("""Town LookUp Table: """, tbl_town)
            st.write("""Town one-hot value is: """, ht_town)
            st.write("""Town one-hot Reference List: """, ref_town)
            st.write("""cc_town list: """, cc_town)
            # flatmodel decoding
            st.write("""FlatModel LookUp Table: """, tbl_flatmodel)
            st.write("""FlatModel one-hot value is : """, ht_fmodel)
            st.write("""FlatModel one-hot Reference List: """, ref_flatmodel)
            st.write("""cc_fmodel list: """, cc_fmodel)
            # final calculation
            ndist_sch = nearest_sch.iloc[0]['ndist']
            ndist_mrt = nearest_mrt.iloc[0]['ndist']
            ndist_fdctr = nearest_fdctr.iloc[0]['ndist']
            # dataframe
            data1=[np.log10(ndist_fdctr), np.log10(floor_area), dist_city, 
                   np.log10(ndist_sch), np.log10(ndist_mrt), cc_storey, cc_ftype, 
                   lease_commence_date]+cc_fmodel+cc_town
            inputs = pd.DataFrame(data=[data1], columns=feature_cols)
            st.write("""final inputs""", inputs)
            
        # PREDICTION
        predictions = model.predict(inputs)
        val = predictions[0]
        log_val =round(10**(val), 2)
        # st.write("""predicted price""", log_val)
        st.success(f"### Predicted resale price of the flat: $ {str(log_val)}")
        st.balloons()
    except:
        st.error("Error: You are missing some parameters, please check your input.")