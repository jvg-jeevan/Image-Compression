class ArithmeticCoder:
    def __init__(self):
        self.FULL = (1 << 32) - 1
        self.HALF = 1 << 31
        self.Q1 = 1 << 30
        self.Q3 = self.Q1 * 3

    def encode(self, data, cumfreq):
        low = 0
        high = self.FULL
        output = []
        pending = 0

        total = cumfreq[-1]

        for symbol in data:
            rng = high - low + 1

            high = low + (rng * cumfreq[symbol + 1] // total) - 1
            low  = low + (rng * cumfreq[symbol]     // total)

            while True:
                if high < self.HALF:
                    output.append(0)
                    output += [1] * pending
                    pending = 0
                elif low >= self.HALF:
                    output.append(1)
                    output += [0] * pending
                    pending = 0
                    low -= self.HALF
                    high -= self.HALF
                elif low >= self.Q1 and high < self.Q3:
                    pending += 1
                    low -= self.Q1
                    high -= self.Q1
                else:
                    break

                low <<= 1
                high = (high << 1) | 1

        pending += 1
        if low < self.Q1:
            output.append(0)
            output += [1] * pending
        else:
            output.append(1)
            output += [0] * pending

        return output

    def decode(self, bits, n, cumfreq):
        total = cumfreq[-1]

        low = 0
        high = self.FULL
        value = 0

        idx = 0
        for _ in range(32):
            value = (value << 1) | (bits[idx] if idx < len(bits) else 0)
            idx += 1

        out = []

        for _ in range(n):
            rng = high - low + 1
            scaled = ((value - low + 1) * total - 1) // rng

            lo = 0
            hi = 255
            while lo < hi:
                mid = (lo + hi) // 2
                if cumfreq[mid + 1] > scaled:
                    hi = mid
                else:
                    lo = mid + 1
            symbol = lo
            out.append(symbol)

            high = low + (rng * cumfreq[symbol + 1] // total) - 1
            low  = low + (rng * cumfreq[symbol]     // total)

            while True:
                if high < self.HALF:
                    pass
                elif low >= self.HALF:
                    low -= self.HALF
                    high -= self.HALF
                    value -= self.HALF
                elif low >= self.Q1 and high < self.Q3:
                    low -= self.Q1
                    high -= self.Q1
                    value -= self.Q1
                else:
                    break

                low <<= 1
                high = (high << 1) | 1

                value <<= 1
                if idx < len(bits):
                    value |= bits[idx]
                idx += 1

        return out