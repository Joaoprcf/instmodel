FROM tensorflow/tensorflow:2.20.0

WORKDIR /app

RUN pip install --upgrade pip && pip install pandas scipy pytest

COPY pyproject.toml README.md LICENSE /app/
COPY instmodel /app/instmodel
RUN pip install --no-deps .

COPY tests/ tests/

CMD ["pytest", "tests", "-v"]
