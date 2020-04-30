import collections
def get_frequency(text_list):
    frequency = collections.defaultdict(int)
    for formula in text_list:
        for sym in formula:
            if sym not in frequency:
                frequency[sym] = 1
            else:
                frequency[sym] += 1
    return frequency

def get_frequncy_counter(text_list):
    temp_list = list()
    for formula in text_list:
        for sym in formula:
            temp_list.append(sym)
    return collections.Counter(temp_list)