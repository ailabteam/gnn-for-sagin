# STEP 1: Chọn một base image chính thức của NVIDIA
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# STEP 2: Thiết lập môi trường và cài đặt các dependency hệ thống
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Tạo symbolic link
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# STEP 3: Thiết lập thư mục làm việc
WORKDIR /app

# STEP 4: Copy file requirements.txt
COPY requirements.txt .

# STEP 5: Cài đặt các thư viện
# TÁCH LÀM 2 BƯỚC ĐỂ GIẢI QUYẾT LỖI INDEX-URL

# Bước 5.1: Cài đặt PyTorch trước tiên, sử dụng --extra-index-url
# Lọc ra các dòng torch từ requirements.txt và cài chúng
#RUN grep 'torch' requirements.txt | pip install --no-cache-dir -r /dev/stdin --extra-index-url https://download.pytorch.org/whl/cu121

# Bước 5.2: Cài đặt các thư viện còn lại từ kho PyPI mặc định
# Lọc bỏ các dòng torch và cài phần còn lại
#RUN grep -v 'torch' requirements.txt | pip install --no-cache-dir -r /dev/stdin
# STEP 5: Install Python libraries from the flexible requirements file
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# STEP 6: Copy toàn bộ mã nguồn của dự án
COPY . .

# STEP 7: Thiết lập biến môi trường
ENV PYTHONUNBUFFERED=1

# STEP 8: Lệnh mặc định
CMD ["python", "-m", "src.train"]
