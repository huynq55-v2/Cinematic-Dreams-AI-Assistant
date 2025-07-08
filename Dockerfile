# Sử dụng hình ảnh (image) Python chính thức làm nền
FROM python:3.10-slim-buster

# Đặt thư mục làm việc bên trong container
WORKDIR /app

# Sao chép tất cả các file từ thư mục hiện tại của bạn vào /app trong container
COPY . /app

# Cài đặt tất cả các gói Python cần thiết từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# MỞ CỔNG 7860: Đây là cổng mà Hugging Face Spaces thường mong đợi.
EXPOSE 7860

# Lệnh sẽ được chạy khi container khởi động.
# Buộc Uvicorn lắng nghe trên cổng $PORT (do HF cung cấp) hoặc 7860.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
