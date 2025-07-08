# Sử dụng hình ảnh (image) Python chính thức làm nền
FROM python:3.10-slim-buster

# Đặt thư mục làm việc bên trong container
WORKDIR /app

# Sao chép tất cả các file từ thư mục hiện tại của bạn vào /app trong container
COPY . /app

# Cài đặt tất cả các gói Python cần thiết từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Mở cổng 8000 để ứng dụng có thể truy cập được từ bên ngoài container
EXPOSE 8000

# Lệnh sẽ được chạy khi container khởi động.
# Nó khởi động server Uvicorn và chạy ứng dụng FastAPI của bạn.
# GOOGLE_API_KEY sẽ được lấy từ biến môi trường của Hugging Face Spaces.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
