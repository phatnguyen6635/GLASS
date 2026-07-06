import os

# Thư mục chứa ảnh
folder_path = r"/home/phatnguyen/Documents/data/glass/20/train/good/"  # đổi thành đường dẫn của bạn

# Lấy danh sách file
files = [
    f for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f))
]

# Sắp xếp theo tên hiện tại
files.sort()

# Đổi tên
for i, filename in enumerate(files, start=1):
    old_path = os.path.join(folder_path, filename)

    # new_filename = f"{i}_{filename}"
    new_filename = f"{filename.split('_')[-1]}"

    new_path = os.path.join(folder_path, new_filename)

    os.rename(old_path, new_path)

print(f"Đã đổi tên {len(files)} file.")