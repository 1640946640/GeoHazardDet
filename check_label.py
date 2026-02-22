# 读取正确的YOLO标签
with open("datasets/labels_from_mask/moxizheng_0.2m_UAV0069.txt") as f:
    content = f.read().strip()

print("Correct YOLO label:")
print(content)
print()

parts = content.split()
print("Label parsing:")
print(f"  class_id = {parts[0]} (0=landslide/mudslide)")
print(f"  x_center = {parts[1]} ({float(parts[1]) * 100:.1f}% of width)")
print(f"  y_center = {parts[2]} ({float(parts[2]) * 100:.1f}% of height)")
print(f"  width = {parts[3]} ({float(parts[3]) * 100:.1f}% of width)")
print(f"  height = {parts[4]} ({float(parts[4]) * 100:.1f}% of height)")
print()

# Calculate bounding box pixel coordinates
w, h = 512, 512
xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
x1 = int((xc - bw / 2) * w)
y1 = int((yc - bh / 2) * h)
x2 = int((xc + bw / 2) * w)
y2 = int((yc + bh / 2) * h)

print("Bounding box pixel coordinates:")
print(f"  X: [{x1}, {x2}] (left side)")
print(f"  Y: [{y1}, {y2}] (upper-middle)")
print()
print("Target location: LEFT side of the image, upper-middle portion")
