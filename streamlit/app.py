import streamlit as st
import requests
import yaml

# Load configuration settings from YAML file
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

st.title("RAG Query App")

user_input = st.text_area("Введите ваш запрос:")

if st.button("Получить ответ"):
    if user_input:
        try:
            retriever_url = f"http://{config['main']['host']}:{config['main']['port']}/query"
            response = requests.post(retriever_url, json={"query": user_input})
            response.raise_for_status()  # Поднимаем исключение для плохих ответов
            result = response.json()
            st.write("Ответ:")
            st.write(result.get("response", "Нет ответа от API"))
        except requests.exceptions.RequestException as e:
            st.error(f"Ошибка при запросе к API: {e}")
    else:
        st.warning("Пожалуйста, введите текст запроса.")
