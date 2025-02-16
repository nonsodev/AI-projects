import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatOpenAI(model="gpt-4")


def generate_restaurant_name_and_items(cuisine):
    restaurant_name_prompt = PromptTemplate(input_variables=["cuisine"],
                                            template="give me a fancy name for an {cuisine} type of restaurant")
    restaurant_menu_items_prompt = PromptTemplate(input_variables=["cuisine"],
                                                  template="give me a popular menu items for an {cuisine} type of "
                                                           "restaurant, only the menu items, nothing else"
                                                           "seperate each menu item with a comma")
    resaurant_name_chain = restaurant_name_prompt | model
    restaurant_menu_items_chain = restaurant_menu_items_prompt | model

    final_chain = RunnableParallel(
        restaurant_name=resaurant_name_chain,
        restaurant_menu_items=restaurant_menu_items_chain
    )

    return final_chain.invoke(cuisine)


st.title("Restaurant Name Generator")
cuisine = st.sidebar.selectbox("Pick a Cuisine", ("Indian", "Italian", "Mexican", "Arabic"))

if cuisine:
    try:
        response = generate_restaurant_name_and_items(cuisine)
        st.header(response["restaurant_name"].content.replace('"', ""))
        menu_items = response["restaurant_menu_items"].content.split(",")

        # menu items
        st.write("**menu items**")
        for item in menu_items:
            st.write("-", item)
    except:
        st.write("**Connection Error**\nConnect to the internet and try again")

# TODO: rewrite except for just the Open ai connection

