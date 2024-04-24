# ==================================================       /     IMPORT LIBRARY    /      =================================================== #
#[model]
import pickle
import base64

#[Data Transformation]
from datetime import datetime
import numpy as np

#[Dashboard]
import streamlit as st
from streamlit_extras.stylable_container import stylable_container


# ==================================================       /     CUSTOMIZATION    /      =================================================== #
# Streamlit Page Configuration
st.set_page_config(
    page_title = "Copper",
    layout = "wide",
    initial_sidebar_state= "expanded"
    )

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_img = get_img_as_base64("Image/gradient2.jpg")

page_bg_img = f"""
<style>
# [data-testid="stAppViewContainer"] {{
# background-image: url("data:image/png;base64,{bg_img}");
# background-size: cover;
# }}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Title
st.title(":red[Copper Sales Prediction:] :orange[A Machine Learning Approach to Status and Price]")


# Options
country = [25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]

status = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM',
       'Wonderful', 'Revised', 'Offered', 'Offerable']

status_dict = {'Lost':0, 'Won':1, 'Draft':2, 'To be approved':3, 'Not lost for AM':4,
                                 'Wonderful':5, 'Revised':6, 'Offered':7, 'Offerable':8}

item_type = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']

item_type_dict = {'W':5, 'WI':6, 'S':3, 'Others':1, 'PL':2, 'IPL':0, 'SLAWR':4}

application = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 
                28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 
                67.0, 68.0, 69.0, 70.0, 79.0, 99.0]

product_ref = [611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 
               164336407, 164337175, 929423819, 1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 
               1665584642, 1665584662, 1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 
               1671876026, 1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]


# Input
tabs = st.tabs(["Predict Sale Status", "Estimate Selling Price"])

with tabs[0]:
    st.write("")
    col1, col2, col3 = st.columns(3, gap="medium") 
    with col1:
        C_country = st.selectbox(label= "Country", options= country, index= 10, key= "country")
        C_itemType = st.selectbox(label= "Item Type", options= item_type, index= 0, key= "item_T")
        C_application = st.selectbox(label= "Application", options= application, index= 4, key= "appl")
        C_prodRef = st.selectbox(label= "Product Reference", options= product_ref, index= 9, key= "prod")
    with col2:
        C_width = st.slider(label= "Width", value= 1500.0, min_value= 700.0, max_value= 1980.0, key= "width")
        C_thick = st.slider(label= "Thickness", min_value= 0.1, max_value= 2500.0, value= 3.64, key= "thick")
        C_itemDate = st.date_input(label= "Item Date", value= datetime(2021, 4, 1), min_value= datetime(2020, 1, 1), max_value= datetime(2022, 1, 1), key= "itemD")
        C_deliveryDate = st.date_input(label= "Delivery Date", value= datetime(2021, 7, 1), min_value= datetime(2020, 1, 2), max_value= datetime(2022, 1, 2), key= "delivery")
    with col3:
        C_customer = st.number_input(label= "Customer-ID", min_value= 12458, max_value= 2147483647, value= 30223403, key= "customer")
        C_quantity = st.number_input(label= "Quantity (ton)", min_value= 0.1, max_value= 1000000000.0, value= 8.71, key= "quantity")
        C_price = st.number_input(label= "Selling Price", min_value= 0.1, max_value= 100001015.0, value= 1253.0, key= "price")
    
    with stylable_container(
        key="red_button",
        css_styles="""
            button {
                background-color: green;
                color: white;
                border-radius: 20px;
                background-image: linear-gradient(90deg, #ff00cc 0%, #333399 100%);
            }
            """,
    ):  
        pred_status_button = st.button("Predict Status")

with tabs[1]:
    st.write("")
    col1, col2  = st.columns(2, gap="medium")
    with col1: 
        label1 = "Country"
        R_country = st.selectbox(f"# {label1}", options= country, index= 10, key= "country1")
        R_status = st.selectbox(label= "Status", options= status, index= 0, key= "status1")
        R_itemType = st.selectbox(label= "Item Type", options= item_type, index= 0, key= "item_T1")
        R_application = st.selectbox(label= "Application", options= application, index= 4, key= "appl1")
        R_prodRef = st.selectbox(label= "Product Reference", options= product_ref, index= 9, key= "prod1")
    with col2:
        R_width = st.slider(label= "Width", value= 1045.0, min_value= 700.0, max_value= 1980.0, key= "width1")
        R_thick = st.slider(label= "Thickness", min_value= 0.1, max_value= 2500.0, value= 3.64, key= "thick1")
        R_customer = st.number_input(label= "Customer-ID", min_value= 12458, max_value= 2147483647, value= 30223403, key= "customer1")
        R_quantity = st.number_input(label= "Quantity (ton)", min_value= 0.1, max_value= 1000000000.0, value= 8.71, key= "quantity1")
        R_itemDate = st.date_input(label= "Item Date", value= datetime(2021, 4, 1), min_value= datetime(2020, 1, 1), max_value= datetime(2022, 1, 1), key= "itemD1")
        R_deliveryDate = st.date_input(label= "Delivery Date", value= datetime(2021, 7, 1), min_value= datetime(2020, 1, 2), max_value= datetime(2022, 1, 2), key= "delivery1")
    
    with stylable_container(
        key="green_button",
        css_styles="""
            button {
                background-color: green;
                color: black;
                border-radius: 20px;
                background-image: linear-gradient(90deg, #89f7fe 0%, #66a6ff 100%);
            }
            """,
    ):
        pred_price_button = st.button("Predict Selling Price")

css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.5rem;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)  


# Format Selling Price
def format_currency(number):
    """Formats a number in Indian currency format with commas and two decimal places.

    Args:
        number: The number to be formatted.

    Returns:
        A string representing the formatted number in Indian currency format.
    """
    number_str = str(number)
    sep = ","
    
    # Check for presence of decimel places
    has_decimal = "." in number_str

    # Integer and decimel parts separation
    if has_decimal:
        integer, decimal = number_str.split(".")
        reversed_num = integer[::-1]
        thousand = reversed_num[:3]
        bal = reversed_num[3:]
        if len(bal) == 0:
            formatted_num = f"₹{thousand[::-1]}.{decimal[:2]}"
        elif len(bal) < 3:
            formatted_int = thousand + "," + bal
            formatted_num = f"₹{formatted_int[::-1]}.{decimal[:2]}"
        else:
            hundreds = sep.join(bal[i:i+2] for i in range(0, len(bal), 2))
            formatted_int = thousand + "," + hundreds
            formatted_num = f"₹{formatted_int[::-1]}.{decimal[:2]}"
    else:
        integer = number_str
        reversed_num = integer[::-1]
        thousand = reversed_num[:3]
        bal = reversed_num[3:]
        if len(bal) == 0:
            formatted_num = f"₹{thousand[::-1]}"
        elif len(bal) < 3:
            formatted_int = thousand + "," + bal
            formatted_num = f"₹{formatted_int[::-1]}"
        else:
            hundreds = sep.join(bal[i:i+2] for i in range(0, len(bal), 2))
            formatted_int = thousand + "," + hundreds
            formatted_num = f"₹{formatted_int[::-1]}"
    return formatted_num

# Model
if pred_status_button:
    # Load pickle model to predict the status.
    with open('Model/classification_model.pkl', 'rb') as f:
        status_model = pickle.load(f)

    # Transform Day, Month, Year
    C_item_year, C_item_month, C_item_day = str(C_itemDate).split("-")
    C_delivery_year, C_delivery_month, C_delivery_day = str(C_deliveryDate).split("-")

    # Combine User Inputs to an array
    user_status_data = np.array([[int(C_customer), 
                                int(C_country),
                                int(item_type_dict.get(C_itemType)),
                                int(C_application),
                                float(C_width),
                                int(C_prodRef),
                                np.log(1 + float(C_quantity)),
                                np.log(1 + float(C_thick)),
                                np.log(1 + float(C_price)),
                                int(C_item_day),
                                int(C_item_month),
                                int(C_item_year), 
                                int(C_delivery_day), 
                                int(C_delivery_month), 
                                int(C_delivery_year)]])
  
    pred_status = status_model.predict(user_status_data)

    # Result
    if pred_status[0] == 1:
        st.success('This sales lead would be successful')
    else:
        st.warning("The lead has been identified as less promising based on the status prediction")

elif pred_price_button:
    # Load pickle model to predict the status.
    with open('Model/regression_model.pkl', 'rb') as f:
        price_model = pickle.load(f)

    # Transform Day, Month, Year
    R_item_year, R_item_month, R_item_day = str(R_itemDate).split("-")
    R_delivery_year, R_delivery_month, R_delivery_day = str(R_deliveryDate).split("-")

    # Combine User Inputs to an array
    user_status_data = np.array([[int(R_customer), 
                                int(R_country),
                                int(status_dict.get(R_status)),
                                int(item_type_dict.get(R_itemType)),
                                int(R_application),
                                float(R_width),
                                int(R_prodRef),
                                np.log(1 + float(R_quantity)),
                                np.log(1 + float(R_thick)),
                                int(R_item_day),
                                int(R_item_month),
                                int(R_item_year), 
                                int(R_delivery_day), 
                                int(R_delivery_month), 
                                int(R_delivery_year)]])
  
    pred_price = price_model.predict(user_status_data)
    final_price = format_currency(round(np.exp(pred_price[0]), 2))

    # Result
    st.info(f"The Estimated Selling Price of the Copper Transaction is {final_price}")



# cd Projects\Project_5\git_project_5\copper-sales-prediction
# streamlit run app.py