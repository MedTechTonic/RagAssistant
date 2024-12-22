import streamlit as st
import requests

st.title("Med RAG")
query_input = st.text_input("Enter your query:")
if st.button("Submit"):
    if query_input:
        try:
            response = requests.post("http://api:8000/query/", json={"query": query_input})
            data = response.json()
            if "response" in data:
                st.subheader("Response")
                st.write(data["response"])
            else:
                st.error("Error: " + data.get("error", "Unknown error"))
        except Exception as e:
            st.error(f"Failed to connect to the backend: {e}")
    else:
        st.warning("Please enter a query.")