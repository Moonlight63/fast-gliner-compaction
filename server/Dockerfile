# CUDA image for the decision server. CPU-only hosts can run app.py from a
# plain virtualenv instead (pip install -r requirements.txt).
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    FGC_HOST=0.0.0.0 \
    FGC_PORT=8765

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py backends.py compat.py needle_test.py ./

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health')"
CMD ["python", "app.py"]
