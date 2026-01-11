def min_max_scale(data):
    """
    Scale data to range 0-1
    Returns: scaled_data, min_val, max_val
    """
    min_val = min(data)
    max_val = max(data)
    scaled = []
    for x in data:
        s = (x - min_val) / (max_val - min_val)
        scaled.append(s)
    return scaled, min_val, max_val


if __name__ == "__main__":
    sizes = [1000, 2000, 3000, 4000, 5000]
    scaled, min_v, max_v = min_max_scale(sizes)
    print("Original:", sizes)
    print("Scaled:", scaled)
    
    # Without scaling - big numbers
    sizes_big = [1000, 2000, 3000, 4000, 5000]
    prices = [200000, 300000, 400000, 500000, 600000]

    # With scaling - 0 to 1
    sizes_scaled, _, _ = min_max_scale(sizes_big)
    prices_scaled, _, _ = min_max_scale(prices)

    print("\nWithout scaling:")
    print("  Sizes:", sizes_big)
    print("  Prices:", prices)

    print("\nWith scaling:")
    print("  Sizes:", sizes_scaled)
    print("  Prices:", prices_scaled)