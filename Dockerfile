# ---------- Base Image ----------
FROM python:3.11.9-slim

# ---------- Set Working Directory ----------
WORKDIR /app

# ---------- Copy Project Files ----------
COPY . /app

# ---------- Upgrade pip ----------
RUN python -m pip install --upgrade pip

# ---------- Install Python Dependencies ----------
RUN pip install -r requirements.txt



# ---------- Expose Streamlit Port ----------
EXPOSE 8501



# ---------- Run the App ----------
CMD ["streamlit", "run", "app.py"]
