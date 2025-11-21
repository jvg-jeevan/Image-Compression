from bitarray import bitarray

def unary_encode(q):
    return bitarray('0' * q + '1')

def unary_decode(bits, index):
    q = 0
    while index < len(bits) and bits[index] == 0:
        q += 1
        index += 1
    index += 1  
    return q, index

def rice_encode(data, k):
    M = 1 << k  
    encoded = bitarray()

    for x in data:
        q = x // M
        r = x % M
        encoded.extend(unary_encode(q))
        encoded.extend(f"{r:0{k}b}")

    return encoded


def rice_decode(encoded_bits, k, length):
    M = 1 << k
    decoded = []
    index = 0

    for _ in range(length):
        q, index = unary_decode(encoded_bits, index)

        r = int(encoded_bits[index:index + k].to01(), 2)
        index += k

        decoded.append(q * M + r)

    return decoded