import streamlit as st
import numpy as np
import pandas as pd
# from reference
from utils import find_postal, find_nearest, dist_from_location, _max_width_, draw_map
import pydeck as pdk
import joblib
import datetime

#==========================================================================================
## CONSTANTS
feature_cols = ['ndist_fd_centre', 'log_area_sqm', 'logndist_school', 'ndist_city', 'logndist_mrt', 'year', 'storey_range', 'flat_type', 'lease_commence_date', 'count_fd_centre', 'count_mrt', 'flat_model_Adjoined', 'flat_model_Common', 'flat_model_DoubleStorey', 'flat_model_Premium', 'region_Central', 'region_East', 'region_North', 'region_North_East', 'region_West', 'ndist_cc_center']



scale_cols = ['ndist_fd_centre','log_area_sqm', 'ndist_cc_center',
              'logndist_school', 'ndist_city','logndist_mrt'] 

ref_flatmodel = ['flat_model_Adjoined', 'flat_model_Common', 
                 'flat_model_DoubleStorey', 'flat_model_Premium']

ref_town = ['region_Central','region_East','region_North','region_North_East','region_West']

townlist = ['-Select Town-', 'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH','CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST','KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES','TOA PAYOH', 'WOODLANDS', 'YISHUN']

fmodellist = ['-Select Model-', 'Model A','Improved','New Generation','Standard', 'Simplified','Apartment','Model A2','Multi Generation', '3Gen','2-room','DBSS','Type S1','Type S2', 'Premium Apartment','Terrace','Premium Apartment Loft', 'Improved-Maisonette','Premium Maisonette','Maisonette', 'Model A-Maisonette','Adjoined flat']

ftypelist = ['-Select Type-', '1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE','MULTI-GENERATION']

storeylist = ['-Select Storey Range-', '01 TO 03','04 TO 06','07 TO 09','10 TO 12', '13 TO 15','16 TO 18', '19 TO 21','22 TO 24','25 TO 27', '28 TO 30','31 TO 33','34 TO 36', '37 TO 39','40 TO 42','43 TO 45', '46 TO 48','49 TO 51']

curr_year = int(datetime.date.today().strftime("%Y"))
lease_yr_list = list(reversed(range(1966, curr_year)))
#==========================================================================================

st.set_page_config(page_title="HDB Resale Price", layout="centered")

#==========================================================================================
# FUNCTIONS

# @st.cache_data
@st.cache_data
def load_data(filepath):
	return pd.read_csv(filepath)


# Map and generate list based on one-hot feature
def matchList(e, key):
	if key==e:
		return 1
	else:
		return 0

    
# Search word return index
def searchInList(targ, slist):
    for i, sname in enumerate(slist):
        # if targ is in sname..
        if targ in sname:
            return i
    return 0

#==========================================================================================
# FLAT/ MAP COORDINATES

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
    
    
def getMapInfo():
    # nearby mrt dict
    mrts = pd.DataFrame.from_dict(nmrt_dict).T  
    mrts = mrts.rename(columns={0: 'flat',
                                1: 'mrt',
                                2: 'mrt_dist',
                                3: 'num_mrt_2km'}).reset_index().drop('index',
                                                                      axis=1)
    mrt2km['type'] = ['MRT']*len(mrt2km)
    # nearby sch dict 
    schools = pd.DataFrame.from_dict(nsch_dict).T   
    schools = schools.rename(columns={0: 'flat',
                                      1: 'school',
                                      2: 'school_dist',
                                      3: 'num_school_2km'}).reset_index().drop('index', 
                                                                               axis=1)
    sch2km['type'] = ['School']*len(sch2km)  
    #  nearby fdctr dict 
    fdctrs = pd.DataFrame.from_dict(nfdctr_dict).T   
    fdctrs = fdctrs.rename(columns={0: 'flat',
                                    1: 'fdctr',
                                    2: 'fdctr_dist',
                                    3: 'num_fdctr_2km'}).reset_index().drop('index', 
                                                                             axis=1)
    fdctr2km['type'] = ['Food Centres']*len(fdctr2km)   
    # nearby cctr 
    ccentre = pd.DataFrame.from_dict(ncctr_dict).T   
    ccentre = ccentre.rename(columns={0: 'flat',
                                      1: 'ccentre',
                                      2: 'ccentre_dist',
                                      3: 'num_ccentre_2km'}).reset_index().drop('index', 
                                                                               axis=1)
    cctr2km['type'] = ['Childcare']*len(cctr2km)
    
    # concat nearby amenites within 2km - stack rows
    map_2km = pd.concat([mrt2km, sch2km, fdctr2km, cctr2km]) 
    map_2km = map_2km.rename(columns={'lat':'LATITUDE', 'lon':'LONGITUDE'})
    # concat all nearby amenities - join cols into 1 row
    sel_flat_coord = pd.concat([flat_coord, 
                                mrts.drop(['flat'], axis=1),
                                schools.drop(['flat'], axis=1),
                                fdctrs.drop(['flat'], axis=1),
                                ccentre.drop(['flat'], axis=1),
                               ], axis=1)
    sel_flat_coord['address'] = sel_flat_coord['BLK_NO'] +" "+sel_flat_coord['ROAD_NAME']
    sel_flat_coord['selected_flat'] = [1]
    # flats dataframe from .csv
    flats = tbl_flats[['LATITUDE','LONGITUDE','block',
                       'street_name','POSTAL','year']].copy()
    flats['address']=flats['block']+" "+flats['street_name']+" "+flats['POSTAL'].map(lambda x: str(x))
    flats = flats.drop_duplicates()
    # take only from 2020 onwards
    flats = flats[flats['year']>=2020]  
    flats = flats.drop(['block','year','POSTAL','street_name'], axis=1)
    flats['selected_flat'] = [0.000001]*len(flats)
    # combine flats with selected flat coord
    flats = pd.concat([flats, sel_flat_coord[['LATITUDE', 
                                    'LONGITUDE', 
                                    'selected_flat', 
                                    'address']] ]).reset_index(drop=True)
    flats[['LATITUDE','LONGITUDE','selected_flat']] = flats[['LATITUDE',
                                                             'LONGITUDE',
                                                             'selected_flat'
                                                            ]].astype(float)
    flats['type'] = ['HDB']*len(flats)
    flats = flats.rename(columns={'address':'name'})
    # combine flats with nearby amenities
    all_bldgs = pd.concat([map_2km, flats]).reset_index()
    #
    cb1, cb2, cb3, cb4, cb5 = st.columns(5)

    show_mrt = cb1.checkbox(label=":orange[MRT Stations]",value=True)  # MRT Stations label_visibility="collapsed"
    show_schools = cb2.checkbox(':green[Schools]',True)
    show_fdctrs = cb3.checkbox(':violet[Food Centres]',True)
    show_cctrs = cb4.checkbox(':blue[Childcare]', True)
    hide_hdb = cb5.checkbox('Hide HDBs',False)    
    #
    amenities_toggle= [show_mrt, show_schools, show_fdctrs, show_cctrs, hide_hdb] 
    # st.write(amenities_toggle)
    draw_map(all_bldgs, 
             float(sel_flat_coord.iloc[0]['LATITUDE']), 
             float(sel_flat_coord.iloc[0]['LONGITUDE']),
             13.5,
             amenities_toggle)
    return all_bldgs

#==========================================================================================
# PROCESS USER INPUT

def getAddrLoc(input_addr):
    flat_coord = getFlatLatLong(input_addr)
    # ---- can't seem to use my distance calculation method -----
    # getNearestDist(flat_coord['LATITUDE'], flat_coord['LONGITUDE'], mrt_coord)
    # get nearby amenities info
    nmrt, nmrt_dict, mrt2km = find_nearest(flat_coord, mrt_coord)
    nsch, nsch_dict, sch2km = find_nearest(flat_coord, school_coord)
    ncctr, ncctr_dict, cctr2km = find_nearest(flat_coord, cctr_coord)  # new
    nfdctr, nfdctr_dict, fdctr2km = find_nearest(flat_coord, fdcentre_coord)
    # dist to city
    city = mrt_coord[mrt_coord['STN_NAME']=='ORCHARD MRT STATION']
    city_loc = (float(city['Latitude']), float(city['Longitude']))
    dist_cty = dist_from_location(flat_coord, city_loc)
    return [flat_coord, 
            nmrt, nmrt_dict, mrt2km, 
            nsch, nsch_dict, sch2km, 
            ncctr, ncctr_dict, cctr2km,
            nfdctr, nfdctr_dict, fdctr2km, 
            dist_cty]


def getAddrInfo(input_addr):  # 560331
    if addrType == "Address":
        inf = ref_flats[ref_flats['address']==input_addr.upper()] 
    else:
        inf = ref_flats[ref_flats['POSTAL']==int(input_addr)]
        
    # display most recent yrs
    inf = inf.sort_values(['year'], ascending=False)
    townidx = searchInList(str(inf['town'].iloc[0]), townlist)
    fm_idx = searchInList(str(inf['flat_model'].iloc[0]), fmodellist)  #flat_model index
    ft_idx = searchInList(str(inf['flat_type'].iloc[0]), ftypelist)
    storey_idx = searchInList(str(inf['storey_range'].iloc[0]), storeylist)

    if inf.empty:
        st.write('-No Past Transactions-')
        lcd_idx = 1
        avgarea = 50
    else:
        lcd_idx = searchInList(str(inf['lease_commence_date'].iloc[0]),
                               list(map(str, lease_yr_list)))  # list of ints to list of strs
        avgarea = int(inf['floor_area_sqm'].max())

    # st.write(inf)
    # st.write ([townidx, fm_idx, ft_idx, lcd_idx, avgarea, storey_idx])  
    return townidx, fm_idx, ft_idx, lcd_idx, avgarea, storey_idx, inf

#==========================================================================================
# DATA PREP FOR INPUT TO MODEL 

def getEncodedRef():
    # encoded vars
    ftype_code = tbl_flattype[tbl_flattype['flat_type']==flat_type].iloc[0]['ref']
    storey_code = tbl_storey[tbl_storey['storey_range']==storey].iloc[0]['ref']
    # one-hot vars
    town_code = tbl_town[tbl_town['town']==town].iloc[0]['region']
    fmodel_code = tbl_flatmodel[tbl_flatmodel['flat_model']==flat_model].iloc[0]['ref']
    town_1hot_list = list(map(lambda e: matchList(e, town_code), ref_town))
    fmodel_1hot_list = list(map(lambda e: matchList(e, fmodel_code), ref_flatmodel))
    return ftype_code, storey_code, town_code, fmodel_code, town_1hot_list, fmodel_1hot_list


def scaleFeatures(scaler, inputs):
    scale_vals = scaler.transform(inputs[scale_cols])
    scale_df = pd.DataFrame(scale_vals, index=inputs.index, columns=scale_cols)
    unscale_df = inputs.drop(scale_cols, axis=1)
    scale_inputs = pd.concat([scale_df, unscale_df], axis=1)
    # drop 'ndist_cc_center'
    scale_inputs = scale_inputs.drop(['ndist_cc_center'], axis=1)
    return scale_inputs

#==========================================================================================

# LOAD AMENITIES COORDINATES
fdcentre_coord = load_data('_datasets/latlong_fd_centres.csv')[['name_of_centre','Latitude','Longitude']]
mrt_coord = load_data('_datasets/latlong_mrt_stns.csv')[['STN_NAME','Latitude','Longitude']]
school_coord = load_data('_datasets/latlong_schools.csv')[['school_name','Latitude','Longitude']]
cctr_coord = load_data('_datasets/latlong_cc_centers.csv')[['centre_name','Latitude','Longitude']]

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
tbl_flats = load_data('_datasets/hdb2/HDBResalePrices2012-2023Jan.csv')

# LOAD MODEL
model = joblib.load('gbr_gs_model.pkl')
scaler = joblib.load('scaler.pkl')


ref_flats = tbl_flats[['block','street_name','storey_range','town','floor_area_sqm','year',
                    'flat_model','flat_type','lease_commence_date','POSTAL','resale_price']].copy()
ref_flats.loc[:, 'address']=ref_flats['block']+" "+ref_flats['street_name']
#====================================================================================
_max_width_(100)

st.title('HDB Resale Price Prediction')
st.write(st.__version__)
# st.sidebar.write(
#     f"This app shows how a Streamlit app can interact easily with a to read or store data."
# )
with st.sidebar:    
    st.subheader('Enter address or postal code')
    addrType = st.radio(' ',('Address', 'Postal Code'), horizontal=True,
                        label_visibility='collapsed')#
    
    # LAYOUT - USER INPUT SECTION 
    if addrType == "Address":
        st.subheader('Input Blk and Address')
        a, b = st.columns(2)   
        blk = a.text_input('Blk','988B')  # 651A
        addr = b.text_input('Address', 'BUANGKOK GREEN') # Ang Mo Kio
        flat_address = blk+" "+addr
    else:
        st.subheader('Input Postal Code') 
        flat_address = st.text_input("Enter Postal Code", '521497')
        
    [flat_coord, 
     nearest_mrt, nmrt_dict, mrt2km, 
     nearest_sch, nsch_dict, sch2km,
     nearest_cctr, ncctr_dict, cctr2km,
     nearest_fdctr, nfdctr_dict, fdctr2km, dist_city] = getAddrLoc(flat_address)

    [townidx, fm_idx, ft_idx, 
     lcd_idx, avgarea, storey_idx, info] = getAddrInfo(flat_address)

    # all_coords = getMapInfo()
    # # st.write(all_coords)

    # a, b = st.columns(2) 
    # search town index from roadname
    # road = flat_coord.iloc[0]['ROAD_NAME'] #.split(' ')[0]  
    # townidx = searchInList(road, townlist) # st.write(road+" : "+str(townidx))
    town = st.selectbox('Town', townlist, index=townidx)
    flat_model = st.selectbox('Flat Model', fmodellist, index=fm_idx)
    flat_type = st.selectbox('Flat Type', ftypelist, index=ft_idx) 
    # get avg. floor area
    # avgarea = int(tbl_avgarea[tbl_avgarea['flat_type']==flat_type].iloc[0]['floor_area_sqm'])
    floor_area = st.slider("Floor Area (sqm)", 30, 300, avgarea, help="""

                         Default value is based on the most recent transactions in
                         the area.

                         If you know the floor area of your apartment, 
                         please adjust accordingly - for a more accurate prediction.

                         """,) 
    storey = st.selectbox('Storey', list(storeylist), index=storey_idx, help="""

                        Default value is based on the most recent transactions in
                        the area.

                        If you know the storey of your apartment, 
                        please adjust accordingly for a more accurate prediction.

                        """,)
    lease_commence_date = st.selectbox('Lease Commencement Date', lease_yr_list, index=lcd_idx)

    # # update info dataframe
    # info = info[info['storey_range']==str(storey)]
    # info = info[info['floor_area_sqm']==int(floor_area)]
    # st.write(info)
    onPress = st.button(label='✨ GIVE ME AN ESTIMATE!')


# update info dataframe - showing most recent yrs
st.subheader('Past Transactions')
info = info[info['storey_range']==str(storey)]
info = info[info['floor_area_sqm']==int(floor_area)]
st.write(info)

st.subheader('Flat Location & Amenities')
st.write("**Please note:** Amenities data are pulled from external sources and may be outdated.")
all_coords = getMapInfo()
# st.write(all_coords)


# OUTPUT SECTION - ON PRESS
if onPress:   
    try:
        ftype_code, storey_code, town_code, fmodel_code, town_1hot_list, fmodel_1hot_list = getEncodedRef()
        
        # DISPLAY RESULTS
        expander = st.expander("BREAKDOWN OF DETAILS")
        with expander:
            st.write("""##### **USER INPUT**""")
            a,b = st.columns(2)
            a.write(f"**TOWN**: {town}")
            b.write(f"**FLAT MODEL**: {flat_model}")
            a.write(f"**FLAT TYPE**: {flat_type}")
            b.write(f"**FLOOR AREA**: {str(floor_area)}")
            a.write(f"**STOREY RANGE**: {storey}", )
            b.write(f"**LEASE COMMENCE YEAR**: {str(lease_commence_date)}")
            st.write("")
            st.write("""##### **GENERATED INFO**""")
            # st.write("""**FLAT LOCATION**: """, flat_coord)
            st.write("""**FLAT ADDRESS**: """, flat_coord.iloc[0, 3])
            st.write("""**DISTANCE TO CITY**: """, str(dist_city)+"km")
            st.write("""**NEAREST MRT**: """, nearest_mrt)
            st.write("""**NEAREST SCHOOL**: """, nearest_sch)
            st.write("""**NEAREST FOOD CENTER**: """, nearest_fdctr)
     
            st.write("""FLAT TYPE *(encoded value to be fed into model)*: """, ftype_code)
            st.write("""STOREY RANGE *(encoded value to be fed into model)*: """, storey_code)
            # town-region decoding
            st.write("")
            a,b = st.columns(2)
            a.write(f"""TOWN LOOKUP TABLE *(for reference)*: """)
            b.write("""TOWN ONE-HOT LIST *(list order for reference)*:""")
            a.write(tbl_town)
            b.write(ref_town)
            st.write("""TOWN one-hot value is: """, town_code)
            st.write("""TOWN *(list to be fed into model)*: """, town_1hot_list)
            # flatmodel decoding
            st.write("")
            a,b = st.columns(2)
            a.write("""FLAT MODEL LOOKUP TABLE *(for reference)*: """)
            b.write("""FLAT MODEL ONE-HOT LIST *(list order for reference)*: """)
            a.write(tbl_flatmodel)
            b.write(ref_flatmodel)
            st.write("""FLAT MODEL one-hot value is : """, fmodel_code)
            st.write("""FLAT MODEL *(list to be fed into model)*: """, fmodel_1hot_list)
            # final calculation
            st.write("")
            st.write("""##### **FINAL INPUT**""")
            ndist_sch = nearest_sch.iloc[0]['ndist']
            ndist_mrt = nearest_mrt.iloc[0]['ndist']
            ndist_fdctr = nearest_fdctr.iloc[0]['ndist']
            ndist_cctr = nearest_cctr.iloc[0]['ndist']
            
            # dataframe
            data1=[np.log10(ndist_fdctr), np.log10(floor_area), np.log10(ndist_sch), 
                   dist_city,np.log10(ndist_mrt), curr_year, storey_code, ftype_code, 
                   lease_commence_date, len(fdctr2km), 
                   len(mrt2km)] + fmodel_1hot_list + town_1hot_list + [ndist_cctr]
            #len(cctr2km),  len(sch2km), 
            inputs = pd.DataFrame(data=[data1], columns=feature_cols)
            st.write("""Before scale""", inputs)
            # run scaler
            scaled_inputs = scaleFeatures(scaler, inputs)
            st.write("""After scale and fed into model""", scaled_inputs)
            
        # PREDICTION  - diff 16,663
        predictions = model.predict(scaled_inputs)
        val = predictions[0]
        log_val =round(10**(val), 2)
        st.success(f"### Predicted resale price of the flat: $ {str(log_val)}")
        st.write("*Note: Predicted resale price is only an estimation and it is dependent on the accuracy of user's input. Predicted resale price has a deviation of +/- 25% from prices indicated in HDB e-service website https://services2.hdb.gov.sg/webapp/BB33RTIS/BB33PReslTrans.jsp*")
        st.balloons()
    except:
        st.error("Error: You are missing some parameters, please check your input.")