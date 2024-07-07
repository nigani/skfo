FROM python:3.10.0
EXPOSE 8501
WORKDIR /bplaner
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD streamlit run app.py