import uuid
import hashlib

# 不可逆
def name_to_uuid_nr(name: str) -> str:
    """将普通字符串转为 0.5.39 版本强制要求的 UUID 格式"""
    hash_obj = hashlib.md5(name.encode('utf-8'))
    return str(uuid.UUID(hash_obj.hexdigest()))


# 可逆
def name_to_uuid_reversible(name: str) -> str:
    """将短字符串可逆地转为 UUID"""
    # 1. 转为 16 进制
    hex_name = name.encode('utf-8').hex()
    # 2. 补齐到 32 位 (UUID 需要 32 个 hex 字符)
    # 使用 '0' 在右侧补齐，或者用特定填充
    padded_hex = hex_name.ljust(32, '0')
    return str(uuid.UUID(padded_hex))

def uuid_to_name_reversible(u_id: str) -> str:
    """从 UUID 还原回字符串"""
    # 1. 去掉连字符
    hex_str = u_id.replace('-', '')
    # 2. 去掉右侧补齐的 '0' 并转回字节
    # 注意：这要求原始明文不能以 hex 编码后的 '0' 结尾，或者记录原始长度
    byte_data = bytes.fromhex(hex_str.rstrip('0'))
    return byte_data.decode('utf-8')


if __name__ == "__main__":
        
    u = name_to_uuid_reversible("user01")
    print(f"UUID: {u}") # 75736572303100000000000000000000
    print(f"还原: {uuid_to_name_reversible(u)}") # user01