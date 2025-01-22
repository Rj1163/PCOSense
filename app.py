import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import About
import faqs
import references
import home

custom_css = """
<style>
body {
    background-color: #f0f2f6;  /* Change background color */
    color: #333333;  /* Change text color */
    font-family: Arial, sans-serif;  /* Change font */
}

/* Style the sidebar */
.sidebar .sidebar-content {
    background-color: #ffffff;  /* Change sidebar background color */
    border-radius: 10px;  /* Add rounded corners to the sidebar */
    padding: 20px;  /* Add padding to the sidebar */
}

/* Style the main content area */
#root .css-17eq0hr {
    background-color: #ffffff;  /* Change main content background color */
    border-radius: 10px;  /* Add rounded corners to the main content area */
    padding: 20px;  /* Add padding to the main content area */
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);  /* Add shadow to the main content area */
}

/* Style the title */
h1 {
    color: #FF69B4;  
}

/* Style the header */
.header {
    background-color: #ffffff;  /* Change header background color */
    padding: 20px;  /* Add padding to the header */
    border-bottom: 1px solid #e0e0e0;  /* Add border to the header */
}

/* Style the footer */
.footer {
    background-color: #ffffff;  /* Change footer background color */
    padding: 20px;  /* Add padding to the footer */
    border-top: 1px solid #e0e0e0;  /* Add border to the footer */
}
</style>
"""

# Inject the custom CSS code using st.markdown()
st.markdown(custom_css, unsafe_allow_html=True)


import streamlit as st

# Define the CSS code to customize the interface
custom_css = """
<style>
/* Your custom CSS styles here */
</style>
"""

# Inject the custom CSS code using st.markdown()
st.markdown(custom_css, unsafe_allow_html=True)

# Load the image
image = Image.open("D:\\3rd Year\\Sem 6\\Minor II\\PCOSense.png")

# Create a grid of columns
col1, col2, col3 = st.columns([1, 8, 1])

# Place the image in the middle column
with col2:
    st.sidebar.image(image, width=10, use_column_width=True)

# Create two columns
col1, col2 = st.columns(2)

# Title
st.title("PCOS Detection Tool")



# To give information
st.info("PCOSense: PCOS detection Tool using ML techniques")

# Sidebar menu
class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        with st.sidebar:
            app = option_menu(
                menu_title='Tabs ',
                options=['Image Tool', 'About', 'Text Prediction', 'References'],
                default_index=1,
                styles={
                    "container": {"padding": "5!important"},
                    "icon": {"color": "white", "font-size": "23px"},
                    "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px",
                                 "--hover-color": "blue"},
                    "nav-link-selected": {"background-color": "#02ab21", "font-weight": "normal"},
                }
            )

        if app == "Image Tool":
            home.app()
        elif app == "About":
            About.app()
        elif app == "Text Prediction":
            faqs.app()
        elif app == 'References':
            references.app()


app = MultiApp()
app.add_app("Image Tool", home.app)
app.add_app("About", About.app)
app.add_app("Text Prediction", faqs.app)
app.add_app("References", references.app)
app.run()

# Add a line indicating the developers' names
st.sidebar.markdown("Developed by: Riddhi Jain and Advaitesha Gupta")
