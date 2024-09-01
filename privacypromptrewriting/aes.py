import base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


def encrypt(message, key):
    # Convert the message from string to bytes
    message_bytes = message.encode()

    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(message_bytes, AES.block_size))
    iv = cipher.iv

    # Combine IV and ciphertext and then encode using Base64
    encrypted_data = base64.b64encode(iv + ct_bytes)
    return encrypted_data.decode('utf-8')  # Convert Base64 bytes to a normal string

def decrypt(ciphertext_str, key):
    # Decode the Base64 encoded string back into bytes
    ciphertext = base64.b64decode(ciphertext_str)

    iv = ciphertext[:16]  # Extract the IV (first 16 bytes)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    pt = unpad(cipher.decrypt(ciphertext[16:]), AES.block_size)

    return pt.decode('utf-8')  # Convert the decrypted bytes back to a string

