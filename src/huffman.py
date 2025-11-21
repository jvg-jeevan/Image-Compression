from heapq import heappush, heappop
from collections import defaultdict
from bitarray import bitarray

class HuffmanNode:
    def __init__(self, symbol=None, freq=None):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freq_dict):
    heap = []
    for symbol, freq in freq_dict.items():
        node = HuffmanNode(symbol, freq)
        heappush(heap, node)

    while len(heap) > 1:
        node1 = heappop(heap)
        node2 = heappop(heap)

        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2

        heappush(heap, merged)

    return heap[0]


def generate_codes(node, prefix="", code_dict=None):
    if code_dict is None:
        code_dict = {}

    if node.symbol is not None:
        code_dict[node.symbol] = prefix
        return code_dict

    generate_codes(node.left, prefix + "0", code_dict)
    generate_codes(node.right, prefix + "1", code_dict)
    return code_dict


def huffman_encode(data, code_dict):
    encoded = bitarray()
    for symbol in data:
        encoded.extend(code_dict[symbol])
    return encoded


def huffman_decode(encoded_bits, root):
    decoded = []
    node = root
    for bit in encoded_bits:
        node = node.left if bit == 0 else node.right
        if node.symbol is not None:
            decoded.append(node.symbol)
            node = root
    return decoded